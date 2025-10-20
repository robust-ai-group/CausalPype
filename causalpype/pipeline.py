from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import networkx as nx
from .steps import Step, STEP_REGISTRY
from .config import PipelineConfig
from pathlib import Path
from causalpype.outputs import OutputCollector
# from causalpype.steps import Step, STEP_REGISTRY
# from causalpype.config import PipelineConfig



class Pipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.steps: List[Step] = []
        self.artifacts: Dict[str, Any] = {}
        self.cohort_id = '' # TODO: Might remove it. Not sure yet!
        self.output_collector = None

    def enable_reporting(self):
        self.config.reporting.enabled = True
        return self

    def disable_reporting(self):
        self.config.reporting.enabled = False
        return self
    
    def report_only(self, *step_names: str):
        self.config.reporting.report_steps = set(step_names)
        self.config.reporting.exclude_steps = None
        return self
    
    def exclude_from_report(self, *step_names):
        self.config.reporting.exclude_steps = set(step_names)
        self.config.reporting.report_steps = None
        return self
    
    def disable_figures(self):
        self.config.reporting.include_figures = False
        return self
    
    def configure_reporting(self, **kwargs):
        """
        Configure reporting options directly.

        Example:
        pipeline.configure_reporting(
            include_data_summary=False,
            include_model_summary=True,
            include_figures=False,
            ...
        )
        """
        for key, value in kwargs.items():
            if hasattr(self.config.reporting, key):
                setattr(self.config.reporting, key, value)
        return self

    def add_step(self, step: Step) -> 'Pipeline':
        """Add a step to the pipeline.

        Args:
            step: A Step instance to add to the pipeline

        Returns:
            Self for method chaining

        Example:
            >>> from causalpype.steps import LoadData, TrainModel
            >>> pipeline.add_step(LoadData(path='data.csv'))
            >>> pipeline.add_step(TrainModel())
        """
        self.steps.append(step)
        return self
    
    def run(self):
        print(f"Starting Pipeline: {self.config.run_id}")

        if self.config.persist and self.config.reporting.enabled:
            output_dir = Path(self.config.output_dir)
            self.output_collector = OutputCollector(
                cohort_id=self.cohort_id,
                run_id=self.config.run_id,
                output_dir=output_dir,
                reporting_config=self.config.reporting
            )

        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            print(f"\n[{i}/{len(self.steps)}] {step.__class__.__name__}")

            should_report = (
                self.output_collector and
                self.config.reporting.should_report_step(step_name)
            )
            if should_report:
                self.output_collector.pipeline_steps.append(step_name)
            outputs = step.execute(self.artifacts, self.config)
            self.artifacts.update(outputs)

        print("\n Pipeline Complete!")
        
        if self.output_collector:
            self._generate_outputs()

        return self.artifacts
    
    def _generate_outputs(self):
        collector = self.output_collector
        reporting = self.config.reporting

        if reporting.include_data_summary:
            if 'data' in self.artifacts:
                data = self.artifacts.get('data')
                collector.record_data_summary(data)
        
        if reporting.include_graph_summary:
            if 'graph' in self.artifacts:
                graph = self.artifacts.get('graph')
                collector.record_graph_summary(graph, self.config.outcome)
        
        if reporting.include_model_summary:
            if 'model' in self.artifacts:
                model = self.artifacts.get('model')
                collector.record_model_summary(model)
        
        if reporting.include_evaluation_metrics:
            if 'evaluation_result' in self.artifacts:
                eval_result = self.artifacts.get('evaluation_result')
                eval_metrics = self.artifacts.get('evakuation_metrics')
                collector.record_evaluation_metrics(eval_result, eval_metrics)

        from causalpype.outputs import CohortMetadata
        from dataclasses import asdict

        cohort_metadata = CohortMetadata(
            cohort_id=self.cohort_id,
            cohort_name=f"Cohort {self.cohort_id}"
        )
        outputs = collector.finalize(
            cohort_metadata=cohort_metadata,
            config=asdict(self.config)
        )
        
        print(f"    Outputs saved to: {collector.save_path / 'results.json'}")
    
    @classmethod
    def from_yaml(cls, path: str):
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        
        pipeline = cls(PipelineConfig(**config.get('config', {})))
        
        for step_cfg in config['steps']:
            step_cls = STEP_REGISTRY[step_cfg['type']]
            pipeline.steps.append(step_cls(**step_cfg.get('params', {})))
        
        return pipeline
    
