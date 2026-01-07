from __future__ import annotations
from typing import Union, Optional, Callable
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
from dowhy import gcm

from causalpype.model import CausalModel
from causalpype.problems.base import BaseProblem
from causalpype.results import RootCauseResult


def _mean_difference(old_samples: np.ndarray, new_samples: np.ndarray) -> float:
    return float(np.mean(new_samples) - np.mean(old_samples))


class RootCause(BaseProblem):
    
    def __init__(
        self,
        model: CausalModel,
        target: str,
        baseline: Union[str, Path, pl.DataFrame, pd.DataFrame],
        comparison: Union[str, Path, pl.DataFrame, pd.DataFrame],
        num_samples: int = 2000,
        method: str = "mean"
    ):
        super().__init__(model)
        self._target = target
        self._baseline = self._load_data(baseline)
        self._comparison = self._load_data(comparison)
        self._num_samples = num_samples
        self._method = method
        self._validate()
    
    def _load_data(self, data: Union[str, Path, pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pl.DataFrame):
            return data.to_pandas()
        else:
            return pl.read_csv(data).to_pandas()
    
    def _validate(self) -> None:
        if self._target not in self.model.nodes:
            raise ValueError(f"Target '{self._target}' not in graph. Available: {self.model.nodes}")
        
        missing_baseline = set(self.model.nodes) - set(self._baseline.columns)
        if missing_baseline:
            raise ValueError(f"Baseline data missing columns: {missing_baseline}")
        
        missing_comparison = set(self.model.nodes) - set(self._comparison.columns)
        if missing_comparison:
            raise ValueError(f"Comparison data missing columns: {missing_comparison}")
    
    def run(self) -> RootCauseResult:
        # Choose difference estimation function
        if self._method == "mean":
            diff_func = _mean_difference
        else:
            diff_func = None  # Use default KL divergence
        
        # Build kwargs
        kwargs = {
            "causal_model": self.model.gcm,
            "old_data": self._baseline,
            "new_data": self._comparison,
            "target_node": self._target,
            "num_samples": self._num_samples,
        }
        if diff_func is not None:
            kwargs["difference_estimation_func"] = diff_func
        
        contributions = gcm.distribution_change(**kwargs)
        
        # Handle different return formats
        contributions_dict = {}
        if contributions is not None:
            for node, value in contributions.items():
                try:
                    val = float(value)
                    # Filter out very small values (numerical noise)
                    if abs(val) > 1e-10:
                        contributions_dict[str(node)] = val
                except (TypeError, ValueError):
                    pass
        
        baseline_mean = float(self._baseline[self._target].mean())
        comparison_mean = float(self._comparison[self._target].mean())
        total_change = comparison_mean - baseline_mean
        
        # If contributions is empty but there is a change, add a placeholder
        if not contributions_dict and abs(total_change) > 1e-6:
            contributions_dict["unexplained"] = total_change
        
        return RootCauseResult(
            target=self._target,
            contributions=contributions_dict,
            baseline_mean=baseline_mean,
            comparison_mean=comparison_mean,
            total_change=total_change,
            metadata={
                "num_samples": self._num_samples,
                "baseline_n": len(self._baseline),
                "comparison_n": len(self._comparison),
                "method": self._method
            }
        )