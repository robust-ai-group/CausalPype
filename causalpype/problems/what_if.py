from __future__ import annotations
from typing import Union, Dict, Any, List, Optional
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
from dowhy import gcm

from causalpype.model import CausalModel
from causalpype.problems.base import BaseProblem
from causalpype.results import WhatIfResult


class WhatIf(BaseProblem):
    """Counterfactual analysis: Given observed data, what would have happened if we intervened?"""
    
    def __init__(
        self,
        model: CausalModel,
        interventions: Dict[str, Any],
        observed_data: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
        num_samples: int = 1000
    ):
        super().__init__(model)
        self._interventions = interventions
        self._num_samples = num_samples
        
        # Use provided data or fall back to training data
        if observed_data is None:
            raise ValueError("observed_data required for counterfactual analysis")
        else:
            self._observed_data = observed_data.to_pandas()
        
        self._validate()
    
    def _validate(self) -> None:
        invalid = [k for k in self._interventions if k not in self.model.nodes]
        if invalid:
            raise ValueError(f"Intervention variables not in graph: {invalid}")
    
    def run(self) -> WhatIfResult:
        intervention_funcs = {
            var: self._constant(val)
            for var, val in self._interventions.items()
        }
        
        # Counterfactual: "What would have happened to THIS data if we intervened?"
        samples = gcm.counterfactual_samples(
            causal_model=self.model.gcm,
            interventions=intervention_funcs,
            observed_data=self._observed_data,
        )
        
        samples_pl = pl.DataFrame(samples)
        
        # Intervention means from counterfactual samples
        intervention_means = {}
        for col in samples_pl.columns:
            try:
                intervention_means[col] = float(samples_pl[col].mean())
            except:
                pass
        
        # Baseline is the actual observed data means
        baseline_means = {}
        for col in self._observed_data.columns:
            try:
                baseline_means[col] = float(self._observed_data[col].mean())
            except:
                pass
        
        # Ensure we have all nodes in both dicts
        all_nodes = set(baseline_means.keys()) | set(intervention_means.keys())
        for node in all_nodes:
            if node not in baseline_means:
                baseline_means[node] = 0.0
            if node not in intervention_means:
                intervention_means[node] = baseline_means.get(node, 0.0)
        
        return WhatIfResult(
            interventions=self._interventions,
            samples=samples_pl,
            baseline_means=baseline_means,
            intervention_means=intervention_means,
            metadata={
                "num_observed": len(self._observed_data),
                "analysis_type": "counterfactual"
            }
        )
    
    @staticmethod
    def _constant(value):
        return lambda x: value


class Intervention(BaseProblem):
    """Interventional analysis: What's the population distribution if we set X=x?"""
    
    def __init__(
        self,
        model: CausalModel,
        interventions: Dict[str, Any],
        num_samples: int = 1000
    ):
        super().__init__(model)
        self._interventions = interventions
        self._num_samples = num_samples
        self._validate()
    
    def _validate(self) -> None:
        invalid = [k for k in self._interventions if k not in self.model.nodes]
        if invalid:
            raise ValueError(f"Intervention variables not in graph: {invalid}")
    
    def run(self) -> WhatIfResult:
        intervention_funcs = {
            var: self._constant(val)
            for var, val in self._interventions.items()
        }
        
        # Interventional: Sample from P(Y | do(X=x))
        samples = gcm.interventional_samples(
            causal_model=self.model.gcm,
            interventions=intervention_funcs,
            num_samples_to_draw=self._num_samples
        )
        
        samples_pl = pl.DataFrame(samples)
        
        intervention_means = {
            col: float(samples_pl[col].mean())
            for col in samples_pl.columns
        }
        
        baseline_means = {}
        if self.model.training_data is not None:
            for col in self.model.training_data.columns:
                baseline_means[col] = float(self.model.training_data[col].mean())
        
        return WhatIfResult(
            interventions=self._interventions,
            samples=samples_pl,
            baseline_means=baseline_means,
            intervention_means=intervention_means,
            metadata={
                "num_samples": self._num_samples,
                "analysis_type": "interventional"
            }
        )
    
    @staticmethod
    def _constant(value):
        return lambda x: value