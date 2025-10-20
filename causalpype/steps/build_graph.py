from dataclasses import dataclass
from causalpype.steps import Step
import polars as pl
import networkx as nx

@dataclass
class BuildGraph(Step):
    """Build a causal graph from an edge list CSV file.

    Attributes:
        edge_list_path: Path to CSV file with edge list (source, target columns)
        graph_key: Key to store graph in artifacts
        output_key: Key to store graph in artifacts
    """
    graph_key = 'graph'
    output_key = 'graph'
    edge_list_path: str

    def execute(self, artifacts, config):
        edge_list = pl.read_csv(self.edge_list_path, has_header=False, new_columns=['source', 'target'])
        graph = nx.from_pandas_edgelist(edge_list, create_using=nx.DiGraph)
        return {self.output_key: graph}