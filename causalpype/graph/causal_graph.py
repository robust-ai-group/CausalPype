from __future__ import annotations
from typing import Union, List, Tuple, Optional, Set
from pathlib import Path

import networkx as nx
import polars as pl
import pandas as pd


class CausalGraph:
    
    def __init__(self, graph: nx.DiGraph):
        self._graph = graph
        self._validate()
    
    @classmethod
    def from_edges(cls, edges: List[Tuple[str, str]]) -> CausalGraph:
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        return cls(graph)
    
    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> CausalGraph:
        df = pl.read_csv(path, has_header=False, new_columns=["source", "target"])
        edges = list(zip(df["source"].to_list(), df["target"].to_list()))
        return cls.from_edges(edges)
    
    @classmethod
    def from_dict(cls, adjacency: dict) -> CausalGraph:
        edges = []
        for source, targets in adjacency.items():
            for target in targets:
                edges.append((source, target))
        return cls.from_edges(edges)
    
    def _validate(self) -> None:
        if not nx.is_directed_acyclic_graph(self._graph):
            cycles = list(nx.simple_cycles(self._graph))
            raise ValueError(f"Graph contains cycles: {cycles}")
        
        if len(self._graph.nodes) == 0:
            raise ValueError("Graph has no nodes")
    
    @property
    def networkx(self) -> nx.DiGraph:
        return self._graph
    
    @property
    def nodes(self) -> List[str]:
        return list(self._graph.nodes)
    
    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self._graph.edges)
    
    @property
    def roots(self) -> List[str]:
        return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]
    
    @property
    def leaves(self) -> List[str]:
        return [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]
    
    def parents(self, node: str) -> List[str]:
        return list(self._graph.predecessors(node))
    
    def children(self, node: str) -> List[str]:
        return list(self._graph.successors(node))
    
    def ancestors(self, node: str) -> Set[str]:
        return nx.ancestors(self._graph, node)
    
    def descendants(self, node: str) -> Set[str]:
        return nx.descendants(self._graph, node)
    
    def topological_order(self) -> List[str]:
        return list(nx.topological_sort(self._graph))
    
    def is_valid_intervention(self, nodes: List[str]) -> bool:
        for node in nodes:
            if node not in self._graph.nodes:
                return False
        return True
    
    def subgraph(self, nodes: List[str]) -> CausalGraph:
        return CausalGraph(self._graph.subgraph(nodes).copy())
    
    def to_csv(self, path: Union[str, Path]) -> None:
        df = pl.DataFrame({
            "source": [e[0] for e in self.edges],
            "target": [e[1] for e in self.edges]
        })
        df.write_csv(path, include_header=False)
    
    def __repr__(self) -> str:
        return f"CausalGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    def __contains__(self, node: str) -> bool:
        return node in self._graph.nodes