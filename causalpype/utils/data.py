from __future__ import annotations
from typing import Union, Optional, List, Tuple, TYPE_CHECKING
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import networkx as nx

if TYPE_CHECKING:
    from causalpype.graph import CausalGraph


class DataLoader:
    
    @staticmethod
    def load(
        path: Union[str, Path],
        null_values: Optional[List[str]] = None
    ) -> pl.DataFrame:
        
        null_values = null_values or ["", " ", "NA", "N/A", "null", "NULL", "None", "?", "#DIV/0!"]
        
        df = pl.read_csv(
            path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=null_values
        )
        
        return df


class GraphAwarePreprocessor:
    
    def __init__(
        self,
        graph: CausalGraph,
        drop_null_threshold: float = 0.99,
        impute_numeric: str = "mean",
        impute_categorical: str = "mode",
        encode_categorical: bool = True
    ):
        self._graph = graph
        self._drop_null_threshold = drop_null_threshold
        self._impute_numeric = impute_numeric
        self._impute_categorical = impute_categorical
        self._encode_categorical = encode_categorical
        self._removed_nodes = []
        self._updated_graph = None
    
    @property
    def removed_nodes(self) -> List[str]:
        return self._removed_nodes
    
    @property
    def updated_graph(self):
        return self._updated_graph
    
    def fit_transform(self, data: pl.DataFrame):
        from causalpype.graph import CausalGraph
        
        graph_nodes = set(self._graph.nodes)
        data_cols = set(data.columns)
        
        keep_cols = [c for c in data.columns if c in graph_nodes]
        df = data.select(keep_cols)
        
        missing_in_data = graph_nodes - data_cols
        self._removed_nodes = list(missing_in_data)
        
        for col in list(df.columns):
            null_frac = df[col].null_count() / len(df)
            if null_frac >= self._drop_null_threshold:
                self._removed_nodes.append(col)
                df = df.drop(col)
        
        df = self._impute(df)
        
        if self._encode_categorical:
            df = self._encode(df)
        
        df = df.drop_nulls()
        df = self._ensure_numeric(df)
        
        final_cols = set(df.columns)
        for node in graph_nodes:
            if node not in final_cols and node not in self._removed_nodes:
                self._removed_nodes.append(node)
        
        new_graph = self._graph.networkx.copy()
        for node in self._removed_nodes:
            if node in new_graph.nodes:
                parents = list(new_graph.predecessors(node))
                children = list(new_graph.successors(node))
                for parent in parents:
                    for child in children:
                        if parent != child and not new_graph.has_edge(parent, child):
                            new_graph.add_edge(parent, child)
                new_graph.remove_node(node)
        
        self._updated_graph = CausalGraph(new_graph)
        
        return df, self._updated_graph
    
    def _impute(self, df: pl.DataFrame) -> pl.DataFrame:
        for col in list(df.columns):
            if df[col].dtype == pl.Object:
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
                except:
                    df = df.drop(col)
                    self._removed_nodes.append(col)
            elif df[col].dtype == pl.Boolean:
                df = df.with_columns(pl.col(col).cast(pl.Int8).alias(col))
        
        exprs = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if dtype.is_numeric():
                if self._impute_numeric == "mean":
                    exprs.append(pl.col(col).fill_null(pl.col(col).mean()))
                elif self._impute_numeric == "median":
                    exprs.append(pl.col(col).fill_null(pl.col(col).median()))
                elif self._impute_numeric == "zero":
                    exprs.append(pl.col(col).fill_null(0))
                else:
                    exprs.append(pl.col(col))
            else:
                if self._impute_categorical == "mode":
                    mode_val = df[col].drop_nulls().mode()
                    if len(mode_val) > 0:
                        exprs.append(pl.col(col).fill_null(mode_val.first()))
                    else:
                        exprs.append(pl.col(col))
                else:
                    exprs.append(pl.col(col))
        
        return df.with_columns(exprs)
    
    def _encode(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if dtype == pl.Utf8 or dtype == pl.Categorical:
                try:
                    numeric = df[col].cast(pl.Float64, strict=False)
                    if numeric.drop_nulls().len() > 0.8 * len(df):
                        exprs.append(numeric.alias(col))
                        continue
                except:
                    pass
                
                unique_vals = df[col].unique().drop_nulls().sort().to_list()
                
                expr = pl.lit(None).cast(pl.Float64)
                for idx, val in enumerate(unique_vals):
                    expr = pl.when(pl.col(col) == val).then(float(idx)).otherwise(expr)
                
                exprs.append(expr.alias(col))
            else:
                exprs.append(pl.col(col))
        
        return df.select(exprs)
    
    def _ensure_numeric(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        
        for col in df.columns:
            if not df[col].dtype.is_numeric():
                try:
                    exprs.append(pl.col(col).cast(pl.Float64))
                except:
                    exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        
        return df.select(exprs)


class Preprocessor:
    
    def __init__(
        self,
        drop_null_threshold: float = 0.5,
        impute_numeric: str = "mean",
        impute_categorical: str = "mode",
        encode_categorical: bool = True,
        drop_remaining_nulls: bool = True
    ):
        self._drop_null_threshold = drop_null_threshold
        self._impute_numeric = impute_numeric
        self._impute_categorical = impute_categorical
        self._encode_categorical = encode_categorical
        self._drop_remaining_nulls = drop_remaining_nulls
    
    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data.clone()
        
        df = self._drop_high_null_columns(df)
        df = self._impute(df)
        
        if self._encode_categorical:
            df = self._encode(df)
        
        if self._drop_remaining_nulls:
            df = df.drop_nulls()
        
        df = self._ensure_numeric(df)
        
        return df
    
    def _drop_high_null_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        cols_to_keep = []
        
        for col in df.columns:
            null_frac = df[col].null_count() / len(df)
            if null_frac <= self._drop_null_threshold:
                cols_to_keep.append(col)
        
        return df.select(cols_to_keep)
    
    def _impute(self, df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if df[col].dtype == pl.Object:
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
                except:
                    df = df.drop(col)
            elif df[col].dtype == pl.Boolean:
                df = df.with_columns(pl.col(col).cast(pl.Int8).alias(col))
        
        exprs = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if dtype.is_numeric():
                if self._impute_numeric == "mean":
                    exprs.append(pl.col(col).fill_null(pl.col(col).mean()))
                elif self._impute_numeric == "median":
                    exprs.append(pl.col(col).fill_null(pl.col(col).median()))
                elif self._impute_numeric == "zero":
                    exprs.append(pl.col(col).fill_null(0))
                else:
                    exprs.append(pl.col(col))
            else:
                if self._impute_categorical == "mode":
                    mode_val = df[col].drop_nulls().mode()
                    if len(mode_val) > 0:
                        exprs.append(pl.col(col).fill_null(mode_val.first()))
                    else:
                        exprs.append(pl.col(col))
                else:
                    exprs.append(pl.col(col))
        
        return df.with_columns(exprs)
    
    def _encode(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if dtype == pl.Utf8 or dtype == pl.Categorical:
                try:
                    numeric = df[col].cast(pl.Float64, strict=False)
                    if numeric.drop_nulls().len() > 0.8 * len(df):
                        exprs.append(numeric.alias(col))
                        continue
                except:
                    pass
                
                unique_vals = df[col].unique().drop_nulls().sort().to_list()
                
                expr = pl.lit(None).cast(pl.Float64)
                for idx, val in enumerate(unique_vals):
                    expr = pl.when(pl.col(col) == val).then(float(idx)).otherwise(expr)
                
                exprs.append(expr.alias(col))
            else:
                exprs.append(pl.col(col))
        
        return df.select(exprs)
    
    def _ensure_numeric(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        
        for col in df.columns:
            if not df[col].dtype.is_numeric():
                try:
                    exprs.append(pl.col(col).cast(pl.Float64))
                except:
                    exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        
        return df.select(exprs)