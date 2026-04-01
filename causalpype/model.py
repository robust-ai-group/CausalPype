import copy
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm


class CausalModel:
    def __init__(self, graph, auto_assign=True, assignment_quality="better"):
        if isinstance(graph, str):
            import pydot
            parsed = pydot.graph_from_dot_data(graph)
            self.graph = nx.nx_pydot.from_pydot(parsed[0])
        elif isinstance(graph, dict):
            self.graph = nx.DiGraph(graph)
        elif isinstance(graph, nx.DiGraph):
            # Deep copy to prevent sharing causal mechanism objects with the
            # caller's graph when nx.DiGraph.copy() (shallow) is used later.
            self.graph = copy.deepcopy(graph)
        else:
            raise ValueError("graph must be a DOT string, networkx DiGraph, or adjacency dict")

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(
                "Graph contains a cycle. CausalPype requires a DAG. "
                f"Cycle: {list(nx.find_cycle(self.graph))}"
            )

        self.scm = gcm.InvertibleStructuralCausalModel(self.graph)
        self.auto_assign = auto_assign
        self._quality_map = {"good": gcm.auto.AssignmentQuality.GOOD,
                             "better": gcm.auto.AssignmentQuality.BETTER,
                             "best": gcm.auto.AssignmentQuality.BEST}
        self.assignment_quality = self._quality_map.get(assignment_quality,
                                                        gcm.auto.AssignmentQuality.BETTER)
        self.data = None
        self._fitted = False

    def fit(self, data):
        missing = set(self.graph.nodes) - set(data.columns)
        if missing:
            raise ValueError(
                f"Data is missing columns for graph nodes: {sorted(missing)}. "
                f"Data columns: {sorted(data.columns)}"
            )
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

    def get_adjustment_set(self, treatment, outcome):
        """Find minimal valid backdoor adjustment set.

        Performs do-surgery (removes outgoing edges from treatment) then
        finds a minimal d-separator in the mutilated graph. This correctly
        identifies variables that block all backdoor paths while preserving
        causal paths.
        """
        if treatment == outcome:
            return []
        g_surgery = self.graph.copy()
        g_surgery.remove_edges_from(list(self.graph.out_edges(treatment)))
        result = nx.algorithms.d_separation.find_minimal_d_separator(
            g_surgery, {treatment}, {outcome}
        )
        if result is None:
            return []
        return [n for n in result if n not in {treatment, outcome}]

    def run(self, tasks, **kwargs):
        from .pipeline import Pipeline
        pipe = Pipeline(self)
        results = pipe.run(tasks, **kwargs)
        self._last_pipeline = pipe
        if not isinstance(tasks, list):
            return results[0]
        return results

    def validate(self):
        """Run validation checks on the fitted causal model."""
        self._check_fitted()
        from .tasks.validate import Validate
        return Validate(method="all").run(self)

    def report(self, format="text"):
        """Generate a report from the last pipeline run.

        Args:
            format: "text" for human-readable output, "dict" for structured data.
        """
        if not hasattr(self, '_last_pipeline') or not self._last_pipeline:
            msg = "No tasks have been run yet."
            if format == "dict":
                return {"error": msg, "graph": None, "data": None, "results": []}
            return msg
        from .report import Report
        r = Report(self, self._last_pipeline.results)
        if format == "dict":
            return r.to_dict()
        return r.to_text()

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