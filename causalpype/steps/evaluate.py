from dataclasses import dataclass
from causalpype.steps.base import Step
from typing import Optional, List, Dict, Any
import networkx as nx
import polars as pl
# import causalpype.pipeline as P


@dataclass
class Evaluate(Step):
    """Evaluate a trained Structural Causal Model.

    Attributes:
        data_key: Key to retrieve input data from artifacts
        model_key: Key to retrieve trained model from artifacts
        output_key: Key to store evaluation results in artifacts
    """
    data_key: str = 'data'
    model_key: str = 'model'
    output_key: str = 'evaluation_result'

    def execute(self, artifacts: Dict[str, Any], config):
        data = artifacts[self.data_key]
        model = artifacts[self.model_key]

        eval_result = model.evaluate(data=data)

        outputs = {
            self.output_key: eval_result
        }

        return outputs