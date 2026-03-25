import numpy as np
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult, _title, _sep, _end, _kv


class SensitivityAnalysisResult(TaskResult):
    def _format(self) -> str:
        d = self.details
        lines = [
            _title("Sensitivity Analysis Results"),
            _kv("Result", self.estimate.upper()),
            _kv("Original ATE", d["original_ate"]),
        ]
        for method_name in ["placebo", "subset", "random_common_cause"]:
            if method_name in d:
                m = d[method_name]
                label = method_name.replace("_", " ").title()
                lines.append(_sep())
                lines.append(f" {label}")
                lines.append(_kv("   Mean Effect", m["mean_effect"]))
                lines.append(_kv("   Std Effect", m["std_effect"]))
                if "p_value" in m:
                    lines.append(_kv("   P-value", m["p_value"]))
                if "fraction" in m:
                    lines.append(_kv("   Fraction", m["fraction"]))
                lines.append(_kv("   Passed", m["passed"]))
        lines.append(_end())
        return "\n".join(lines)


class SensitivityAnalysis(BaseTask):
    """Test robustness of causal effect estimates via refutation methods.

    Runs three tests on the GCM-based ATE estimate:
    - Placebo treatment: permute treatment
    - Data subset: re-estimate on random subsets
    - Random common cause: inject random confounder
    """
    name = "sensitivity_analysis"

    def __init__(self, treatment, outcome, treatment_value=1, control_value=0,
                 num_simulations=10, subset_fraction=0.8,
                 methods=None, num_samples=2000):
        self.treatment = treatment
        self.outcome = outcome
        self.treatment_value = treatment_value
        self.control_value = control_value
        self.num_simulations = num_simulations
        self.subset_fraction = subset_fraction
        self.methods = methods or ["placebo", "subset", "random_common_cause"]
        self.num_samples = num_samples

    def _rebuild_and_fit(self, graph, data, quality):
        scm = gcm.InvertibleStructuralCausalModel(graph)
        gcm.auto.assign_causal_mechanisms(scm, data, quality=quality)
        gcm.fit(scm, data)
        return scm

    def _estimate_ate(self, scm):
        return float(gcm.average_causal_effect(
            scm,
            target_node=self.outcome,
            interventions_alternative={self.treatment: lambda x: self.treatment_value},
            interventions_reference={self.treatment: lambda x: self.control_value},
            num_samples_to_draw=self.num_samples,
        ))

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.treatment)
        self._check_node(model, self.outcome)

        rng = np.random.default_rng(42)
        original_ate = self._estimate_ate(model.scm)
        results = {"original_ate": original_ate}

        # Placebo: permute treatment column, re-fit, re-estimate
        if "placebo" in self.methods:
            effects = []
            for _ in range(self.num_simulations):
                data_placebo = model.data.copy()
                data_placebo[self.treatment] = rng.permutation(
                    data_placebo[self.treatment].values
                )
                scm_p = self._rebuild_and_fit(
                    model.graph, data_placebo, model.assignment_quality
                )
                effects.append(self._estimate_ate(scm_p))
            effects = np.array(effects)
            p_value = float(np.mean(np.abs(effects) >= np.abs(original_ate)))
            results["placebo"] = {
                "mean_effect": float(np.mean(effects)),
                "std_effect": float(np.std(effects)),
                "p_value": p_value,
                "passed": p_value < 0.05,
            }

        # Subset: re-estimate on random subsets
        if "subset" in self.methods:
            effects = []
            for _ in range(self.num_simulations):
                data_sub = model.data.sample(
                    frac=self.subset_fraction, random_state=int(rng.integers(1e9))
                )
                scm_s = self._rebuild_and_fit(
                    model.graph, data_sub, model.assignment_quality
                )
                effects.append(self._estimate_ate(scm_s))
            effects = np.array(effects)
            results["subset"] = {
                "mean_effect": float(np.mean(effects)),
                "std_effect": float(np.std(effects)),
                "fraction": self.subset_fraction,
                "passed": bool(np.abs(np.mean(effects) - original_ate)
                               < 2 * np.std(effects) + 1e-10),
            }

        # Random common cause: add random confounder to graph
        if "random_common_cause" in self.methods:
            effects = []
            for _ in range(self.num_simulations):
                data_rcc = model.data.copy()
                data_rcc["_random_cause"] = rng.standard_normal(len(data_rcc))
                graph_rcc = model.graph.copy()
                graph_rcc.add_edges_from([
                    ("_random_cause", self.treatment),
                    ("_random_cause", self.outcome),
                ])
                scm_r = self._rebuild_and_fit(
                    graph_rcc, data_rcc, model.assignment_quality
                )
                effects.append(self._estimate_ate(scm_r))
            effects = np.array(effects)
            results["random_common_cause"] = {
                "mean_effect": float(np.mean(effects)),
                "std_effect": float(np.std(effects)),
                "original_ate": original_ate,
                "passed": bool(np.abs(np.mean(effects) - original_ate)
                               < 2 * np.std(effects) + 1e-10),
            }

        # Overall assessment
        tests_run = [k for k in self.methods if k in results]
        tests_passed = [k for k in tests_run if results[k].get("passed", False)]
        all_passed = len(tests_passed) == len(tests_run)

        return SensitivityAnalysisResult(
            task_name="Sensitivity Analysis",
            estimate="robust" if all_passed else "sensitive",
            details=results,
        )
