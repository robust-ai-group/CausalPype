import polars as pl
from causalpype import (
    CausalEngine,
    CausalGraph,
    CausalModel,
    GraphAwarePreprocessor,
    ReportConfig,
    ReportGenerator,
)

GRAPH_PATH = "~/datasets/better4u/edge_list - causal_graph.csv"
DATA_PATH = "~/datasets/better4u/HUABB_data_cleaned.csv"

print("=" * 70)
print("CausalPype - Health Dataset Analysis")
print("=" * 70)

print("\n[1] Loading graph...")
graph = CausalGraph.from_csv(GRAPH_PATH)
print(f"    {len(graph.nodes)} nodes, {len(graph.edges)} edges")

print("\n[2] Loading data...")
data = pl.read_csv(DATA_PATH, infer_schema_length=10000, ignore_errors=True)
print(f"    Shape: {data.shape}")

print("\n[3] Preprocessing (graph-aware)...")
preprocessor = GraphAwarePreprocessor(
    graph=graph,
    drop_null_threshold=0.99,
    impute_numeric="mean",
    encode_categorical=True
)
data_clean, updated_graph = preprocessor.fit_transform(data)

print(f"    Removed nodes: {len(preprocessor.removed_nodes)}")
print(f"    Updated graph: {len(updated_graph.nodes)} nodes, {len(updated_graph.edges)} edges")
print(f"    Clean data shape: {data_clean.shape}")

available_nodes = updated_graph.nodes
print(f"    Available nodes: {available_nodes}")

print("\n[4] Fitting causal model...")
model = CausalModel(updated_graph)

engine = CausalEngine(model)

desc = engine.describe(data=data_clean)
print(desc.summary())
print(desc.warnings)

model.fit(data_clean, quality="good")


desc = engine.describe()
print(desc.node_stats)

print(f"    {engine}")

results = []

print("\n[5] Running analyses...")

treatment_candidates = ["MET_week", "PAL", "sleep_hours", "sitting_day"]
outcome_candidates = ["bw", "whr"]

treatment = next((t for t in treatment_candidates if t in available_nodes), None)
outcome = next((o for o in outcome_candidates if o in available_nodes), None)

if treatment and outcome:
    print(f"\n--- Treatment Effect: {treatment} -> {outcome} ---")
    treatment_mean = data_clean[treatment].mean()
    treatment_std = data_clean[treatment].std()
    
    te_result = engine.treatment_effect(
        treatment=treatment,
        outcome=outcome,
        treatment_value=float(treatment_mean + treatment_std),
        control_value=float(treatment_mean - treatment_std),
        num_samples=1000
    )
    print(te_result.summary())
    results.append(te_result)

intervention_candidates = {
    "MET_week": 20.0,
    "PAL": 2.0,
    "sleep_hours": 8.0,
    "fruits_day": 3.0,
    "fish_wk": 3.0,
    "nuts_seeds_day": 1.0,
    "legumes_week": 3.0,
}

interventions = {k: v for k, v in intervention_candidates.items() if k in available_nodes}

if interventions:
    print(f"\n--- What-If: {interventions} ---")
    whatif_result = engine.what_if(interventions=interventions, num_samples=1000)
    print(whatif_result.summary())
    results.append(whatif_result)

target_candidates = ["bw", "whr"]
target = next((t for t in target_candidates if t in available_nodes), None)

if target:
    print(f"\n--- Root Cause: {target} ---")
    n = len(data_clean)
    baseline = data_clean[:n//2]
    comparison = data_clean[n//2:]
    
    rc_result = engine.root_cause(
        target=target,
        baseline=baseline,
        comparison=comparison,
        num_samples=500
    )
    print(rc_result.summary())
    results.append(rc_result)

print("\n[6] Generating report...")
engine.generate_report(
    results=results,
    output_path="causal_analysis_report.md",
    title="Health Data Causal Analysis",
    author="CausalPype",
    include_plots=True,
    include_tables=True
)
print("    Report saved to: causal_analysis_report.md")

print("\n" + "=" * 70)
print("Complete!")
print("=" * 70)

