from __future__ import annotations
from typing import Union, Dict, Any, Optional, List, Tuple
from pathlib import Path

import polars as pl
import pandas as pd

from causalpype.graph import CausalGraph
from causalpype.model import CausalModel
from causalpype.problems import RootCause, TreatmentEffect, WhatIf, Intervention
from causalpype.results import RootCauseResult, TreatmentEffectResult, WhatIfResult, DescribeResult
from causalpype.validation import Validator
from causalpype.utils import DataLoader, Preprocessor
from causalpype.reports import ReportGenerator, ReportConfig


class CausalEngine:
    
    def __init__(self, model: CausalModel):
        self._model = model
        self._validator = Validator(model)
    
    @classmethod
    def from_edges(
        cls,
        edges: List[Tuple[str, str]],
        data: Optional[Union[str, Path, pl.DataFrame, pd.DataFrame]] = None,
        fit_quality: str = "better"
    ) -> CausalEngine:
        
        model = CausalModel.from_edges(edges)
        
        if data is not None:
            model.fit(data, quality=fit_quality)
        
        return cls(model)
    
    @classmethod
    def from_csv(
        cls,
        graph_path: Union[str, Path],
        data_path: Optional[Union[str, Path]] = None,
        fit_quality: str = "better"
    ) -> CausalEngine:
        
        model = CausalModel.from_csv(graph_path)
        
        if data_path is not None:
            model.fit(data_path, quality=fit_quality)
        
        return cls(model)
    
    @classmethod
    def from_dict(
        cls,
        adjacency: Dict[str, List[str]],
        data: Optional[Union[str, Path, pl.DataFrame, pd.DataFrame]] = None,
        fit_quality: str = "better"
    ) -> CausalEngine:
        
        model = CausalModel.from_dict(adjacency)
        
        if data is not None:
            model.fit(data, quality=fit_quality)
        
        return cls(model)
    
    @property
    def model(self) -> CausalModel:
        return self._model
    
    @property
    def graph(self) -> CausalGraph:
        return self._model.graph
    
    @property
    def nodes(self) -> List[str]:
        return self._model.nodes
    
    @property
    def is_fitted(self) -> bool:
        return self._model.is_fitted
    
    def fit(
        self,
        data: Union[str, Path, pl.DataFrame, pd.DataFrame],
        quality: str = "better"
    ) -> CausalEngine:
        
        self._model.fit(data, quality=quality)
        return self
    
    def root_cause(
        self,
        target: str,
        baseline: Union[str, Path, pl.DataFrame, pd.DataFrame],
        comparison: Union[str, Path, pl.DataFrame, pd.DataFrame],
        num_samples: int = 2000
    ) -> RootCauseResult:
        
        analysis = RootCause(
            model=self._model,
            target=target,
            baseline=baseline,
            comparison=comparison,
            num_samples=num_samples
        )
        
        return analysis.run()
    
    def treatment_effect(
        self,
        treatment: str,
        outcome: str,
        data: Optional[Union[str, Path, pl.DataFrame, pd.DataFrame]] = None,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
        num_samples: int = 1000
    ) -> TreatmentEffectResult:
        
        analysis = TreatmentEffect(
            model=self._model,
            treatment=treatment,
            outcome=outcome,
            data=data,
            treatment_value=treatment_value,
            control_value=control_value,
            num_samples=num_samples
        )
        
        return analysis.run()
    
    def what_if(
        self,
        interventions: Dict[str, Any],
        observed_data: Optional[Union[str, Path, pl.DataFrame, pd.DataFrame]] = None,
    ) -> WhatIfResult:
        """Counterfactual analysis: What would have happened to observed data if we intervened?
        """
        analysis = WhatIf(
            model=self._model,
            interventions=interventions,
            observed_data=observed_data,
        )
        
        return analysis.run()
    
    def intervene(
        self,
        interventions: Dict[str, Any],
        num_samples: int = 1000
    ) -> WhatIfResult:
        """Interventional analysis: Sample from P(Y | do(X=x)).
        """
        analysis = Intervention(
            model=self._model,
            interventions=interventions,
            num_samples=num_samples
        )
        
        return analysis.run()
    
    def validate(self, result):
        return self._validator.run_all(result)
    
    def sample(self, n: int = 1000) -> pl.DataFrame:
        return self._model.sample(n)
    
    def evaluate(self, data=None):
        return self._model.evaluate(data)
    
    def describe(self, data: Optional[Union[str, Path, pl.DataFrame, pd.DataFrame]] = None) -> DescribeResult:
        graph = self._model.graph
        
        warnings = []
        node_stats = {}
        
        if data is not None:
            if isinstance(data, (str, Path)):
                df = pl.read_csv(data)
            elif isinstance(data, pd.DataFrame):
                df = pl.DataFrame(data)
            else:
                df = data
            
            data_cols = set(df.columns)
            graph_nodes = set(graph.nodes)
            
            missing_in_data = graph_nodes - data_cols
            if missing_in_data:
                warnings.append(f"Graph nodes missing in data: {missing_in_data}")
            
            extra_in_data = data_cols - graph_nodes
            if extra_in_data:
                warnings.append(f"Data columns not in graph (will be ignored): {len(extra_in_data)} columns")
            
            for node in graph.nodes:
                if node in df.columns:
                    col = df[node]
                    null_count = col.null_count()
                    null_pct = null_count / len(df)
                    
                    if null_pct >= 0.99:
                        warnings.append(f"Node '{node}' has {null_pct:.1%} nulls - will be removed")
                    elif null_pct > 0.5:
                        warnings.append(f"Node '{node}' has {null_pct:.1%} nulls - high missing rate")
                    
                    if col.dtype.is_numeric():
                        non_null = col.drop_nulls()
                        if len(non_null) > 0:
                            node_stats[node] = {
                                "mean": float(non_null.mean()),
                                "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                                "min": float(non_null.min()),
                                "max": float(non_null.max()),
                                "null_pct": float(null_pct)
                            }
        elif self._model.training_data is not None:
            df = self._model.training_data
            for node in graph.nodes:
                if node in df.columns:
                    col = df[node]
                    node_stats[node] = {
                        "mean": float(col.mean()),
                        "std": float(col.std()) if len(col) > 1 else 0.0,
                        "min": float(col.min()),
                        "max": float(col.max()),
                        "null_pct": 0.0
                    }
        
        training_rows = 0
        training_cols = 0
        if self._model.training_data is not None:
            training_rows = len(self._model.training_data)
            training_cols = len(self._model.training_data.columns)
        
        return DescribeResult(
            n_nodes=len(graph.nodes),
            n_edges=len(graph.edges),
            nodes=graph.nodes,
            root_nodes=graph.roots,
            leaf_nodes=graph.leaves,
            is_fitted=self._model.is_fitted,
            has_training_data=self._model.training_data is not None,
            training_data_rows=training_rows,
            training_data_cols=training_cols,
            node_stats=node_stats,
            warnings=warnings
        )
    
    def generate_report(
        self,
        results: List[Any],
        output_path: Union[str, Path],
        title: str = "Causal Analysis Report",
        author: str = "",
        include_plots: bool = True,
        include_tables: bool = True,
        include_metadata: bool = True
    ) -> None:
        
        config = ReportConfig(
            title=title,
            author=author,
            include_plots=include_plots,
            include_tables=include_tables,
            include_metadata=include_metadata
        )
        
        generator = ReportGenerator(config)
        generator.generate(results, output_path, title)
    
    def save(self, path: Union[str, Path], include_data: bool = False) -> None:
        self._model.save(path, include_data=include_data)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> CausalEngine:
        model = CausalModel.load(path)
        return cls(model)
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"CausalEngine(nodes={len(self.nodes)}, {status})"


def load_data(path: Union[str, Path]) -> pl.DataFrame:
    return DataLoader.load(path)


def preprocess(
    data: pl.DataFrame,
    drop_null_threshold: float = 0.5,
    impute_numeric: str = "mean",
    encode_categorical: bool = True
) -> pl.DataFrame:
    
    preprocessor = Preprocessor(
        drop_null_threshold=drop_null_threshold,
        impute_numeric=impute_numeric,
        encode_categorical=encode_categorical
    )
    
    return preprocessor.fit_transform(data)