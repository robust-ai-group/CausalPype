import dowhy.gcm as gcm
from dowhy.gcm.validation import RejectionResult
from .base import BaseTask, TaskResult


class Validate(BaseTask):
    """Validate causal model assumptions using DoWhy GCM refutation methods.

    Note: Structure validation runs multiple conditional independence tests
    without multiple testing correction. For graphs with many edges, consider
    using a lower significance_level (e.g. 0.01) to reduce false rejections.
    """
    name = "validate"

    def __init__(self, method="all", significance_level=0.05):
        self.method = method
        self.significance_level = significance_level

    def run(self, model, **kwargs):
        self.validate(model)
        results = {}
        all_passed = True

        if self.method in ("structure", "all"):
            rejection, details = gcm.refute_causal_structure(
                model.graph, model.data,
                significance_level=self.significance_level,
            )
            structure_passed = rejection == RejectionResult.NOT_REJECTED
            all_passed &= structure_passed

            node_summaries = {}
            flat_edge_tests = {}
            for node, tests in details.items():
                node_summary = {}
                edge_tests = tests.get("edge_dependence_test", {})
                for parent, result in edge_tests.items():
                    edge_key = f"{parent} -> {node}"
                    edge_info = {
                        "p_value": result.get("p_value"),
                        "success": result.get("success"),
                    }
                    node_summary[edge_key] = edge_info
                    flat_edge_tests[edge_key] = edge_info
                lm_test = tests.get("local_markov_test", {})
                if lm_test:
                    node_summary["local_markov"] = {
                        "p_value": lm_test.get("p_value"),
                        "success": lm_test.get("success"),
                    }
                if node_summary:
                    node_summaries[node] = node_summary

            n_tests = len(flat_edge_tests) + sum(
                1 for ns in node_summaries.values() if "local_markov" in ns
            )

            results["structure"] = {
                "passed": structure_passed,
                "edge_tests": flat_edge_tests,
                "node_details": node_summaries,
                "n_tests": n_tests,
                "bonferroni_level": self.significance_level / n_tests if n_tests > 0 else self.significance_level,
            }

        if self.method in ("model", "all"):
            model_rejection = gcm.refute_invertible_model(
                model.scm, model.data,
                significance_level=self.significance_level,
            )
            model_passed = model_rejection == RejectionResult.NOT_REJECTED
            all_passed &= model_passed
            results["model"] = {
                "passed": model_passed,
                "result": model_rejection.name,
            }

        return TaskResult(
            task_name="Validation",
            estimate="passed" if all_passed else "issues_found",
            details=results,
        )
