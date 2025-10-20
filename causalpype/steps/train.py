from dataclasses import dataclass
from causalpype.steps.base import Step
from typing import Optional
import networkx as nx


@dataclass
class TrainModel(Step):
    """Train a Structural Causal Model.

    Attributes:
        data_key: Key to retrieve input data from artifacts
        graph_key: Key to retrieve causal graph from artifacts
        outcome: Target variable name (uses config.outcome if empty)
        output_key: Key to store trained model in artifacts
    """
    data_key: str = 'data'
    graph_key = 'graph'
    outcome: str = ''
    output_key: str = 'model'

    def execute(self, artifacts, config):
        data = artifacts[self.data_key]
        graph = artifacts[self.graph_key]
        outcome = self.outcome or config.outcome

        from causalpype.models import CausalModel
        model = CausalModel(graph=graph, outcome=self.outcome)
        model.assign_mechanisms(dataset=data)
        model.fit(data)

        return {self.output_key: model}
    
