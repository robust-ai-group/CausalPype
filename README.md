# CausalPype

A small Python library for causal analysis with structural causal models,
built on top of [DoWhy-GCM](https://github.com/py-why/dowhy) and
[EconML](https://github.com/py-why/EconML).

CausalPype is a thin wrapper that gives DoWhy-GCM and EconML a single
pipeline API. You describe a causal graph, fit a structural causal model
on your data, and then run any subset of 14 composable causal tasks on
the same fitted model in one call. The library does not implement new
causal methods; it just makes using the existing ones together less
painful.

## Install

CausalPype requires Python 3.12 or newer. For now, install from GitHub:

```bash
pip install git+https://github.com/HUA-RobustAI/CausalPype.git
```

### Manual installation

```
git clone git@github.com:robust-ai-group/CausalPype.git
cd CausalPype
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync
```


## Quickstart

```python
import pandas as pd
from causalpype import CausalModel, ATE, ArrowStrength

# 1. A causal graph as an adjacency dict (parent -> list of children).
dag = {
    "age":   ["sysBP", "TenYearCHD"],
    "sysBP": ["TenYearCHD"],
}

# 2. Fit a structural causal model on your data.
df = pd.read_csv("examples/data/framingham.csv")
model = CausalModel(dag).fit(df)

# 3. Run any number of causal tasks on the fitted model.
results = model.run([
    ATE(treatment="sysBP", outcome="TenYearCHD",
        treatment_value=140, control_value=120),
    ArrowStrength(target="TenYearCHD"),
])

for r in results:
    print(r)
```

`CausalModel` also accepts a `networkx.DiGraph` directly if you prefer
to build the graph that way.

## The 14 tasks

| Category | Task | What it answers |
|---|---|---|
| Treatment Effects | `ATE` | Average causal effect of a fixed treatment change |
| | `CATE` | Heterogeneous effects via EconML (linear DML, causal forest, metalearner) |
| | `KNNIntervention` | Non parametric ATE / ATT / ATC via nearest neighbour matching |
| Interventional / Counterfactual | `Intervention` | Hard do intervention on one or more nodes |
| | `StochasticIntervention` | Shift the treatment distribution and measure the downstream effect |
| | `Counterfactual` | Individual level counterfactual outcomes |
| | `CausalEffectCurve` | Sweep a treatment over its range and plot the dose response |
| Mechanistic | `ArrowStrength` | Strength of each direct parent of a target node |
| | `IntrinsicCausalInfluence` | Shapley value variance decomposition over the full graph |
| Anomaly / Shift | `AnomalyAttribution` | Attribute anomalous observations to upstream root causes |
| | `DistributionChange` | Attribute a distributional shift between two datasets to upstream nodes |
| Fairness & Validation | `FairnessAudit` | Counterfactual fairness across a protected attribute |
| | `Validate` | Refutation tests for DAG structure and SCM fit |
| | `SensitivityAnalysis` | Placebo, subset, and random confounder robustness checks |

Every task returns a `TaskResult` with a scalar `estimate`, a `details`
dict, and a `to_dict()` method for JSON serialisation.

## License

CausalPype is released under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full text.
