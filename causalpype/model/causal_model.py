from __future__ import annotations
from typing import Union, Optional, Dict, Any
from pathlib import Path
import pickle

import polars as pl
import pandas as pd
import numpy as np
from dowhy import gcm

from causalpype.graph import CausalGraph


class CausalModel:
    
    def __init__(self, graph: CausalGraph):
        self._graph = graph
        self._gcm = gcm.InvertibleStructuralCausalModel(graph.networkx)
        self._fitted = False
        self._training_data: Optional[pl.DataFrame] = None
    
    @classmethod
    def from_edges(cls, edges) -> CausalModel:
        graph = CausalGraph.from_edges(edges)
        return cls(graph)
    
    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> CausalModel:
        graph = CausalGraph.from_csv(path)
        return cls(graph)
    
    @classmethod
    def from_dict(cls, adjacency: dict) -> CausalModel:
        graph = CausalGraph.from_dict(adjacency)
        return cls(graph)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> CausalModel:
        path = Path(path)
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        graph = CausalGraph(state["graph"])
        model = cls(graph)
        model._gcm = state["gcm"]
        model._fitted = state["fitted"]
        
        if state.get("training_data") is not None:
            model._training_data = pl.DataFrame(state["training_data"])
        
        return model
    
    @property
    def graph(self) -> CausalGraph:
        return self._graph
    
    @property
    def gcm(self) -> gcm.InvertibleStructuralCausalModel:
        return self._gcm
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    @property
    def nodes(self) -> list:
        return self._graph.nodes
    
    @property
    def training_data(self) -> Optional[pl.DataFrame]:
        return self._training_data
    
    def fit(
        self,
        data: Union[str, Path, pl.DataFrame, pd.DataFrame],
        quality: str = "better"
    ) -> CausalModel:
        
        df = self._load_data(data)
        self._validate_data(df)
        
        pandas_df = df.to_pandas()
        
        quality_enum = getattr(gcm.auto.AssignmentQuality, quality.upper())
        gcm.auto.assign_causal_mechanisms(
            causal_model=self._gcm,
            based_on=pandas_df,
            quality=quality_enum,
            override_models=True
        )
        
        gcm.fit(self._gcm, pandas_df)
        
        self._fitted = True
        self._training_data = df
        
        return self
    
    def save(self, path: Union[str, Path], include_data: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "graph": self._graph.networkx,
            "gcm": self._gcm,
            "fitted": self._fitted,
            "training_data": self._training_data.to_pandas() if include_data and self._training_data is not None else None
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def sample(self, n: int = 1000) -> pl.DataFrame:
        self._check_fitted()
        samples = gcm.draw_samples(self._gcm, num_samples=n)
        return pl.DataFrame(samples)
    
    def evaluate(self, data: Optional[Union[str, Path, pl.DataFrame, pd.DataFrame]] = None) -> Dict[str, Any]:
        self._check_fitted()
        
        if data is None:
            if self._training_data is None:
                raise ValueError("No data provided and no training data available")
            df = self._training_data
        else:
            df = self._load_data(data)
        
        result = gcm.evaluate_causal_model(
            causal_model=self._gcm,
            data=df.to_pandas(),
            evaluate_causal_mechanisms=True,
            compare_mechanism_baselines=True,
            evaluate_invertibility_assumptions=True,
            evaluate_overall_kl_divergence=True,
            evaluate_causal_structure=True
        )
        
        return result
    
    def _load_data(self, data: Union[str, Path, pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, pd.DataFrame):
            return pl.DataFrame(data)
        else:
            return pl.read_csv(data)
    
    def _validate_data(self, data: pl.DataFrame) -> None:
        missing = set(self._graph.nodes) - set(data.columns)
        if missing:
            raise ValueError(f"Data missing columns for graph nodes: {missing}")
    
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model must be fitted first. Call .fit(data)")
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"CausalModel(nodes={len(self.nodes)}, {status})"