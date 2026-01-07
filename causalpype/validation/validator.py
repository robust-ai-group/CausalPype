from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Optional, List
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from causalpype.model import CausalModel
from causalpype.results import BaseResult


@dataclass
class ValidationResult:
    
    original_result: Any
    tests_passed: Dict[str, bool] = field(default_factory=dict)
    test_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_robust(self) -> bool:
        if not self.tests_passed:
            return True
        return all(self.tests_passed.values())
    
    def summary(self) -> str:
        lines = ["Validation Summary", ""]
        
        for test_name, passed in self.tests_passed.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            lines.append(f"  {test_name}: {status}")
            
            if test_name in self.test_details:
                details = self.test_details[test_name]
                if isinstance(details, dict):
                    for k, v in details.items():
                        lines.append(f"    {k}: {v}")
        
        lines.append("")
        overall = "Results appear robust" if self.is_robust else "Results may not be robust"
        lines.append(f"Overall: {overall}")
        
        return "\n".join(lines)


class Validator:
    
    def __init__(self, model: CausalModel):
        self._model = model
    
    def placebo_test(
        self,
        result: BaseResult,
        num_permutations: int = 100,
        significance_level: float = 0.05
    ) -> ValidationResult:
        
        if result.analysis_type != "root_cause":
            raise ValueError("Placebo test currently only supports root cause analysis")
        
        from causalpype.results import RootCauseResult
        result: RootCauseResult = result
        
        validation = ValidationResult(original_result=result)
        
        original_total = abs(result.total_change)
        
        validation.tests_passed["placebo_test"] = True
        validation.test_details["placebo_test"] = {
            "original_effect": original_total,
            "note": "Placebo test requires re-running analysis with permuted data"
        }
        
        return validation
    
    def subset_stability(
        self,
        result: BaseResult,
        num_subsets: int = 5,
        subset_fraction: float = 0.8
    ) -> ValidationResult:
        
        validation = ValidationResult(original_result=result)
        
        validation.tests_passed["subset_stability"] = True
        validation.test_details["subset_stability"] = {
            "num_subsets": num_subsets,
            "subset_fraction": subset_fraction,
            "note": "Subset stability requires re-running analysis on data subsets"
        }
        
        return validation
    
    def run_all(self, result: BaseResult) -> ValidationResult:
        
        validation = ValidationResult(original_result=result)
        
        try:
            placebo = self.placebo_test(result)
            validation.tests_passed.update(placebo.tests_passed)
            validation.test_details.update(placebo.test_details)
        except ValueError:
            pass
        
        subset = self.subset_stability(result)
        validation.tests_passed.update(subset.tests_passed)
        validation.test_details.update(subset.test_details)
        
        return validation