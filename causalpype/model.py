import pandas as pd
import networkx as nx
import dowhy.gcm as gcm


class CausalModel:
    def __init__(self, graph, auto_assign=True, assignment_quality="good"):
        if isinstance(graph, str):
            import pydot
            parsed = pydot.graph_from_dot_data(graph)
            self.graph = nx.nx_pydot.from_pydot(parsed[0])
        elif isinstance(graph, dict):
            self.graph = nx.DiGraph(graph)
        elif isinstance(graph, nx.DiGraph):
            self.graph = graph
        else:
            raise ValueError("graph must be a DOT string, networkx DiGraph, or adjacency dict")

        self.scm = gcm.InvertibleStructuralCausalModel(self.graph)
        self.auto_assign = auto_assign
        self._quality_map = {"good": gcm.auto.AssignmentQuality.GOOD,
                             "better": gcm.auto.AssignmentQuality.BETTER,
                             "best": gcm.auto.AssignmentQuality.BEST}
        self.assignment_quality = self._quality_map.get(assignment_quality,
                                                        gcm.auto.AssignmentQuality.GOOD)
        self.data = None
        self._fitted = False

    def fit(self, data):
        self.data = data
        if self.auto_assign:
            gcm.auto.assign_causal_mechanisms(self.scm, data, quality=self.assignment_quality)
        gcm.fit(self.scm, data)
        self._fitted = True
        return self

    def set_mechanism(self, node, mechanism):
        self.scm.set_causal_mechanism(node, mechanism)
        return self

    def draw_samples(self, n=1000):
        self._check_fitted()
        return gcm.draw_samples(self.scm, n)

    def get_parents(self, node):
        return list(self.graph.predecessors(node))

    def get_children(self, node):
        return list(self.graph.successors(node))

    def get_roots(self):
        return [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]

    def get_all_paths(self, source, target):
        return list(nx.all_simple_paths(self.graph, source, target))

    def run(self, tasks, **kwargs):
        from .pipeline import Pipeline
        pipe = Pipeline(self)
        result = pipe.run(tasks, **kwargs)
        self._last_pipeline = pipe
        return result

    def summary(self):
        if not hasattr(self, '_last_pipeline') or not self._last_pipeline:
            return "No tasks have been run yet."
        return self._last_pipeline.summary()

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call model.fit(data) first.")

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return (f"CausalModel({status}, nodes={list(self.graph.nodes)}, "
                f"edges={list(self.graph.edges)})")