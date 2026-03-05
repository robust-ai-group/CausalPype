import numpy as np
import pandas as pd
import dowhy.gcm as gcm
from .base import BaseTask, TaskResult


class AnomalyAttribution(BaseTask):
    """For anomalous observations, attribute the anomaly score to upstream nodes."""
    name = "anomaly_attribution"

    def __init__(self, target, anomaly_data=None, anomaly_threshold_percentile=95):
        self.target = target
        self.anomaly_data = anomaly_data
        self.anomaly_threshold_percentile = anomaly_threshold_percentile

    def run(self, model, **kwargs):
        self.validate(model)
        self._check_node(model, self.target)

        if self.anomaly_data is not None:
            anomaly_samples = self.anomaly_data
        else:
            # Auto-detect anomalies using anomaly scores
            scores = gcm.anomaly_scores(model.scm, model.data)
            target_scores = scores[self.target]
            threshold = np.percentile(target_scores, self.anomaly_threshold_percentile)
            mask = target_scores >= threshold
            anomaly_samples = model.data[mask]

        if len(anomaly_samples) == 0:
            return TaskResult(
                task_name="Anomaly Attribution",
                estimate={},
                details={
                    "target": self.target,
                    "error": "No anomalies found above the threshold",
                    "n_anomalies": 0,
                    "mean_attributions": {},
                },
            )

        attributions = gcm.attribute_anomalies(
            model.scm,
            target_node=self.target,
            anomaly_samples=anomaly_samples,
        )

        mean_attributions = {str(k): float(np.mean(v)) for k, v in attributions.items()}

        return TaskResult(
            task_name="Anomaly Attribution",
            estimate=mean_attributions,
            details={
                "target": self.target,
                "mean_attributions": mean_attributions,
                "n_anomalies": len(anomaly_samples),
                "raw_attributions": attributions,
            },
        )