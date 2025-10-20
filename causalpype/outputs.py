from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import pickle
from datetime import datetime
import hashlib
from causalpype.config import ReportingConfig


@dataclass
class CohortMetadata:
    """Metadata identifying the cohort/site."""
    cohort_id: str
    cohort_name: str
    institution: Optional[str] = None
    country: Optional[str] = None
    run_date: str = None
    pipeline_version: str = None

    def __post_init__(self):
        if self.run_date is None:
            self.run_date = datetime.now().isoformat()


@dataclass
class DataSummary:
    """Privacy-preserving data summary (no raw data!)."""
    n_samples: int
    n_features: int
    feature_names: List[str]

    # Aggregated statistics only
    feature_types: Dict[str, str]  # feature -> 'numeric'/'categorical'/'binary'
    missing_rates: Dict[str, float]  # feature -> proportion missing

    # Distribution summaries (not raw data)
    numeric_stats: Dict[str, Dict[str, float]]  # feature -> {mean, std, min, max, quartiles}
    categorical_counts: Dict[str, int]  # feature -> n_unique_values

    # Outcome-specific
    outcome_variable: str
    outcome_type: str
    outcome_distribution: Dict[str, float]  # summary stats

    # Data quality indicators
    data_quality_score: float  # 0-1 score
    quality_flags: List[str]  # warnings/issues


@dataclass
class GraphSummary:
    """Causal graph structure summary."""
    n_nodes: int
    n_edges: int
    node_names: List[str]
    edge_list: List[tuple]  # [(source, target), ...]

    # Graph properties
    is_dag: bool
    n_connected_components: int
    avg_degree: float
    max_degree: int

    # Outcome-specific
    outcome_variable: str
    n_parents_of_outcome: int
    parents_of_outcome: List[str]
    n_ancestors_of_outcome: int

    # Graph hash for comparison
    graph_hash: str


@dataclass
class ModelSummary:
    """Trained model summary (no model weights!)."""
    model_type: str
    target_variable: str

    # Training info
    n_training_samples: int
    n_features_used: int
    features_used: List[str]
    training_time_seconds: float

    # Mechanism assignments (what model for each variable)
    causal_mechanisms: Dict[str, str]  # variable -> mechanism_type

    # Model parameters (aggregated, not full weights)
    n_parameters: int
    mechanism_complexities: Dict[str, int]  # variable -> n_params


@dataclass
class EvaluationMetrics:
    """Standardized evaluation metrics for cross-cohort comparison."""

    # DoWhy SCM evaluation
    overall_kl_divergence: Optional[float] = None

    # Per-mechanism performance
    per_variable_r2: Dict[str, float] = None  # variable -> R²
    per_variable_mse: Dict[str, float] = None  # variable -> MSE
    per_variable_mae: Dict[str, float] = None  # variable -> MAE

    # Outcome prediction performance
    outcome_r2: Optional[float] = None
    outcome_rmse: Optional[float] = None
    outcome_mae: Optional[float] = None

    # Cross-validation metrics (if applicable)
    cv_mean_r2: Optional[float] = None
    cv_std_r2: Optional[float] = None

    # Model comparison (if multiple models trained)
    model_rankings: Optional[Dict[str, float]] = None

    # Causal structure validation
    graph_structure_score: Optional[float] = None


@dataclass
class PipelineOutputs:
    """Complete standardized outputs from a pipeline run."""

    # Identification
    cohort_metadata: CohortMetadata
    run_id: str

    # Summaries (privacy-preserving)
    data_summary: DataSummary
    # graph_summary: GraphSummary
    model_summary: ModelSummary
    evaluation_metrics: EvaluationMetrics

    # Execution metadata
    execution_time_seconds: float
    pipeline_steps: List[str]
    pipeline_config: Dict[str, Any]

    # Artifacts for local review
    figure_paths: Dict[str, str]  # name -> relative_path
    log_file_path: str

    # Serialization metadata
    output_version: str = "1.0.0"
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, output_dir: Path):
        """Save outputs in standardized format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results as JSON
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save Python-serialized version for programmatic access
        pickle_path = output_dir / 'results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"✅ Saved standardized outputs to: {output_dir}")
        return results_path

    @classmethod
    def load(cls, output_dir: Path) -> 'PipelineOutputs':
        """Load outputs from directory."""
        pickle_path = Path(output_dir) / 'results.pkl'
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
          

class OutputCollector:
    def __init__(self, cohort_id, run_id, output_dir, reporting_config: ReportingConfig):
        self.cohort_id = cohort_id
        self.run_id = run_id
        self.output_dir = output_dir
        self.reporting_config = reporting_config
        self.start_time = datetime.now()

        self.data_info = {}
        self.graph_info = {}
        self.model_info = {}
        self.eval_info = {}
        self.figures = {}
        self.pipeline_steps = []
        self.config = {}

        self.save_path = self.output_dir / self.run_id

    
    def record_data_summary(self, data, codebook_metadata=None):
        import polars as pl
        self.data_info = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'feature_names': data.columns,
            'feature_types': {},
            'missing_rates': {},
            'numeric_stats': {},
            'categorical_counts': {}
        }

        for col in data.columns:
            if data[col].dtype == pl.Boolean:
                self.data_info['feature_types'][col] = 'binary'
            elif data[col].dtype.is_numeric():
                self.data_info['feature_types'][col] = 'numeric'
            else:
                self.data_info['feature_types'][col] = 'categorical'
            
            null_count = data[col].null_count()
            self.data_info['missing_rates'][col] = null_count / len(data)

            if data[col].dtype.is_numeric():
                print(col)
                self.data_info['numeric_stats'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'q25': float(data[col].quantile(0.25)),
                    'q50': float(data[col].quantile(0.50)),
                    'q75': float(data[col].quantile(0.75)),
                }
            else:
                self.data_info['categorical_counts'][col] = data[col].n_unique()

    def record_graph_summary(self, graph, outcome):
        import networkx as nx
        edge_list_str = str(sorted(graph.edges()))
        self.graph_info = {
            'edge_list': list(graph.edges())
        }
    
    def record_model_summary(self, model):
        from causalpype.models.causal_model import CausalModel

        self.model_info = {
            'outcome': model.outcome,
            'n_features_used': len(model.features()),
            'features_used': model.features(),
            'causal_mechanisms': {},
        }

        if hasattr(model, 'causal_mechanisms') and model.causal_mechanisms:
            # for node, mechanism in model.causal_mechanisms.items():
            #     self.model_info['causal_mechanisms'][node] = type(mechanism).__name__
            # TODO
            pass

    def record_evaluation_metrics(self, eval_result, eval_metrics=None):
        self.eval_info = {
            'overall_kl_divergence': None,
            'per_variable_r2': {},
            'per_variable_mse': {}
        }
        # TODO:
        if eval_metrics:
            self.eval_info.update(eval_metrics)

    
    def finalize(self, cohort_metadata, config):
        execution_time = (datetime.now() - self.start_time).total_seconds()

        outcome = config.get('outcome')
        outcome_stats = self.data_info['numeric_stats'].get(outcome, {})

        data_summary = DataSummary(
            n_samples=self.data_info.get('n_samples', 0),
            n_features=self.data_info.get('n_features', 0),
            feature_names=self.data_info.get('feature_names', []),
            feature_types=self.data_info.get('feature_types', {}),
            missing_rates=self.data_info.get('missing_rates', {}),
            numeric_stats=self.data_info.get('numeric_stats', {}),
            categorical_counts=self.data_info.get('categorical_counts', {}),
            outcome_variable=outcome,
            outcome_type=self.data_info.get('feature_types', {}).get(outcome, 'unknown'),
            outcome_distribution=outcome_stats,
            data_quality_score=1.0 - sum(self.data_info.get('missing_rates', {}).values()) / max(len(self.data_info.get('missing_rates', {})), 1),
            quality_flags=[]
        )

        # Build graph summary
        # graph_summary = GraphSummary(**self.graph_info)

        # Build model summary
        model_summary = ModelSummary(
            model_type=self.model_info.get('model_type', 'unknown'),
            target_variable=self.model_info.get('target_variable', outcome),
            n_training_samples=self.data_info.get('n_samples', 0),
            n_features_used=self.model_info.get('n_features_used', 0),
            features_used=self.model_info.get('features_used', []),
            training_time_seconds=self.model_info.get('training_time_seconds', 0),
            causal_mechanisms=self.model_info.get('causal_mechanisms', {}),
            n_parameters=0,  # Could extract if needed
            mechanism_complexities=self.model_info.get('mechanism_complexities', {})
        )

        # Build evaluation metrics
        eval_metrics = EvaluationMetrics(**self.eval_info)

        # Create final outputs
        outputs = PipelineOutputs(
            cohort_metadata=cohort_metadata,
            run_id=self.run_id,
            data_summary=data_summary,
            # graph_summary=graph_summary,
            model_summary=model_summary,
            evaluation_metrics=eval_metrics,
            execution_time_seconds=execution_time,
            pipeline_steps=self.pipeline_steps,
            pipeline_config=config,
            figure_paths=self.figures,
            log_file_path='pipeline.log'
        )

        # Save
        outputs.save(self.save_path)

        return outputs

    
    def record_figure(self, name, path):
        if self.reporting_config.should_report_figure(name):
            self.figures[name] = str(Path(path).relative_to(self.save_path))
