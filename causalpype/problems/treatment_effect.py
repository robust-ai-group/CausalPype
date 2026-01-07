from __future__ import annotations
from typing import Union, Optional, List
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
from dowhy import gcm

from causalpype.model import CausalModel
from causalpype.problems.base import BaseProblem
from causalpype.results import TreatmentEffectResult


class TreatmentEffect(BaseProblem):
    
    def __init__(
        self,
        model: CausalModel,
        treatment: str,
        outcome: str,
        data: Optional[Union[str, Path, pl.DataFrame, pd.DataFrame]] = None,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
        num_samples: int = 1000
    ):
        super().__init__(model)
        self._treatment = treatment
        self._outcome = outcome
        self._treatment_value = treatment_value
        self._control_value = control_value
        self._num_samples = num_samples
        
        if data is not None:
            self._data = self._load_data(data)
        elif model.training_data is not None:
            self._data = model.training_data.to_pandas()
        else:
            raise ValueError("No data provided and model has no training data")
        
        self._validate()
    
    def _load_data(self, data: Union[str, Path, pl.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pl.DataFrame):
            return data.to_pandas()
        else:
            return pl.read_csv(data).to_pandas()
    
    def _validate(self) -> None:
        if self._treatment not in self.model.nodes:
            raise ValueError(f"Treatment '{self._treatment}' not in graph")
        
        if self._outcome not in self.model.nodes:
            raise ValueError(f"Outcome '{self._outcome}' not in graph")
    
    def run(self) -> TreatmentEffectResult:
        def set_treatment(x):
            return self._treatment_value
        
        def set_control(x):
            return self._control_value
        
        treated_samples = gcm.interventional_samples(
            causal_model=self.model.gcm,
            interventions={self._treatment: set_treatment},
            num_samples_to_draw=self._num_samples
        )
        
        control_samples = gcm.interventional_samples(
            causal_model=self.model.gcm,
            interventions={self._treatment: set_control},
            num_samples_to_draw=self._num_samples
        )
        
        treated_mean = float(treated_samples[self._outcome].mean())
        control_mean = float(control_samples[self._outcome].mean())
        ate = treated_mean - control_mean
        
        return TreatmentEffectResult(
            treatment=self._treatment,
            outcome=self._outcome,
            ate=ate,
            treated_mean=treated_mean,
            control_mean=control_mean,
            n_treated=self._num_samples,
            n_control=self._num_samples,
            metadata={
                "treatment_value": self._treatment_value,
                "control_value": self._control_value,
                "num_samples": self._num_samples
            }
        )