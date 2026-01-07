from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from causalpype.results import BaseResult
from causalpype.model import CausalModel


class BaseProblem(ABC):
    
    def __init__(self, model: CausalModel):
        if not model.is_fitted:
            raise RuntimeError("Model must be fitted before running analysis")
        self._model = model
    
    @property
    def model(self) -> CausalModel:
        return self._model
    
    @abstractmethod
    def run(self) -> BaseResult:
        pass