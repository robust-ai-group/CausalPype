import dowhy.gcm as gcm
from .base import BaseTask, TaskResult, _title, _sep, _end, _kv


class IntrinsicCausalInfluenceResult(TaskResult):
    def _format(self) -> str:
        d = self.details
        lines = [
            _title("Intrinsic Causal Influence Results"),
            _kv("Target", d["target"]),
            _kv("Total Variance Explained", d["total_variance_explained"]),
            _sep(),
        ]
        influences = d["influences"]
        normalized = d["normalized"]
        for k, v in sorted(influences.items(), key=lambda x: abs(x[1]), reverse=True):
            pct = f"({normalized[k]:.1%})"
            lines.append(_kv(f" {k} {pct}", v))
        lines.append(_end())
        return "\n".join(lines)


class IntrinsicCausalInfluence(BaseTask):
    """Attribute variance in a target to upstream noise terms via Shapley values."""
    name = "intrinsic_causal_influence"

    def __init__(self, target, prediction_model="approx"):
        self.target = target
        self.prediction_model = prediction_model

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.target)

        influences = gcm.intrinsic_causal_influence(
            model.scm,
            target_node=self.target,
            prediction_model=self.prediction_model,
        )

        influences_clean = {str(k): float(v) for k, v in influences.items()}

        total = sum(influences_clean.values())
        normalized = {k: v / total if total > 0 else 0 for k, v in influences_clean.items()}

        return IntrinsicCausalInfluenceResult(
            task_name="Intrinsic Causal Influence",
            estimate=influences_clean,
            details={
                "target": self.target,
                "influences": influences_clean,
                "normalized": normalized,
                "total_variance_explained": total,
            },
        )