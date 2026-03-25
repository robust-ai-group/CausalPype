from typing import List, Union

from .tasks.base import TaskResult


# Keys to skip in text output (large arrays, internal objects)
_SKIP_IN_TEXT = {
    "samples", "individual_effects", "predictions", "residuals",
    "anomalies", "estimator", "cate_model", "policy",
    "individual_counterfactuals", "all_ite", "ite_treated",
    "ite_control", "response_df", "assignments",
    "counterfactual_samples", "interventional_samples",
    "noise_data", "observed_data", "raw_attributions", "raw_strengths",
    "individual_nde", "individual_nie", "lower_bound", "upper_bound",
}


class Report:
    """Generate human-readable or structured reports from causal analysis results."""

    def __init__(self, model, results: Union[TaskResult, List[TaskResult]]):
        self.model = model
        self.results = results if isinstance(results, list) else [results]

    def to_text(self) -> str:
        """Generate a human-readable plain text report."""
        sections = [
            self._header(),
            self._graph_summary(),
            self._data_summary(),
        ]
        for result in self.results:
            sections.append(self._format_result(result))
        sections.append("=" * 60)
        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        """Structured dict — ready for JSON serialization, future GUI/API/LLM."""
        return {
            "graph": {
                "nodes": list(self.model.graph.nodes),
                "edges": [(u, v) for u, v in self.model.graph.edges],
                "n_nodes": self.model.graph.number_of_nodes(),
                "n_edges": self.model.graph.number_of_edges(),
            },
            "data": {
                "n_rows": len(self.model.data) if self.model.data is not None else 0,
                "columns": list(self.model.data.columns) if self.model.data is not None else [],
            },
            "results": [r.to_dict() for r in self.results],
        }

    def _header(self) -> str:
        return "\n".join([
            "=" * 60,
            "CausalPype Analysis Report",
            "=" * 60,
        ])

    def _graph_summary(self) -> str:
        g = self.model.graph
        roots = [n for n in g.nodes if g.in_degree(n) == 0]
        leaves = [n for n in g.nodes if g.out_degree(n) == 0]
        lines = [
            "--- Causal Graph ---",
            f"  Nodes ({g.number_of_nodes()}): {list(g.nodes)}",
            f"  Edges ({g.number_of_edges()}): {[(u, v) for u, v in g.edges]}",
            f"  Root nodes: {roots}",
            f"  Leaf nodes: {leaves}",
        ]
        return "\n".join(lines)

    def _data_summary(self) -> str:
        if self.model.data is None:
            return "--- Data ---\n  No data loaded."
        data = self.model.data
        lines = [
            "--- Data ---",
            f"  Rows: {len(data)}",
            f"  Columns: {len(data.columns)}",
        ]
        return "\n".join(lines)

    def _format_result(self, result: TaskResult) -> str:
        return result._format()
