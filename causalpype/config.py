from dataclasses import dataclass, field
from typing import Optional, Set

@dataclass
class ReportingConfig:
    enabled = True
    include_data_summary = True
    include_graph_summary = True
    include_model_summary = True
    include_evaluation_metrics = True
    include_figures = True
    include_logs = True

    report_steps: Optional[Set[str]] = None
    exclude_steps: Optional[Set[str]] = None

    def should_report_step(self, step_name: str):
        if not self.enabled:
            return False
        
        if self.report_steps is not None:
            return step_name in self.report_steps
        
        if self.exclude_steps is not None:
            return step_name not in self.exclude_steps
    
        return True
    
    def should_report_figure(self, figure_name):
        if not self.enabled or not self.include_figures:
            return False
        
        return True
    

@dataclass
class PipelineConfig:
    run_id: str = "run"
    output_dir: str = "results"
    persist: bool = True
    outcome: Optional[str] = None
    random_seed: int = 2025

    reporting: ReportingConfig = field(default_factory=ReportingConfig)