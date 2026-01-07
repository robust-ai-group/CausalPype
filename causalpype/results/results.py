from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

import polars as pl


@dataclass
class BaseResult:
    
    analysis_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def to_json(self, path: Optional[Path] = None) -> str:
        data = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(data)
        return data
    
    def summary(self) -> str:
        raise NotImplementedError


@dataclass
class DescribeResult:
    
    n_nodes: int = 0
    n_edges: int = 0
    nodes: List[str] = field(default_factory=list)
    root_nodes: List[str] = field(default_factory=list)
    leaf_nodes: List[str] = field(default_factory=list)
    is_fitted: bool = False
    has_training_data: bool = False
    training_data_rows: int = 0
    training_data_cols: int = 0
    node_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": {
                "n_nodes": self.n_nodes,
                "n_edges": self.n_edges,
                "nodes": self.nodes,
                "root_nodes": self.root_nodes,
                "leaf_nodes": self.leaf_nodes,
            },
            "model": {
                "is_fitted": self.is_fitted,
                "has_training_data": self.has_training_data,
                "training_data_rows": self.training_data_rows,
                "training_data_cols": self.training_data_cols,
            },
            "node_stats": self.node_stats,
            "warnings": self.warnings,
        }
    
    def to_json(self, path: Optional[Path] = None) -> str:
        data = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(data)
        return data
    
    def summary(self) -> str:
        lines = [
            "CausalEngine Description",
            "=" * 40,
            "",
            "Graph:",
            f"  Nodes: {self.n_nodes}",
            f"  Edges: {self.n_edges}",
            f"  Roots: {self.root_nodes}",
            f"  Leaves: {self.leaf_nodes}",
            "",
            "Model:",
            f"  Fitted: {self.is_fitted}",
            f"  Training data: {self.training_data_rows} rows x {self.training_data_cols} cols" if self.has_training_data else "  Training data: None",
        ]
        
        if self.node_stats:
            lines.append("")
            lines.append("Node Statistics:")
            for node, stats in list(self.node_stats.items())[:10]:
                lines.append(f"  {node}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, nulls={stats['null_pct']:.1%}")
            if len(self.node_stats) > 10:
                lines.append(f"  ... and {len(self.node_stats) - 10} more nodes")
        
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        
        return "\n".join(lines)


@dataclass
class RootCauseResult(BaseResult):
    
    target: str = ""
    contributions: Dict[str, float] = field(default_factory=dict)
    baseline_mean: float = 0.0
    comparison_mean: float = 0.0
    total_change: float = 0.0
    analysis_type: str = "root_cause"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "target": self.target,
            "contributions": self.contributions,
            "baseline_mean": self.baseline_mean,
            "comparison_mean": self.comparison_mean,
            "total_change": self.total_change,
            "metadata": self.metadata
        }
    
    def summary(self) -> str:
        direction = "increased" if self.total_change > 0 else "decreased"
        pct_change = abs(self.total_change / self.baseline_mean * 100) if self.baseline_mean != 0 else 0
        
        lines = [
            f"Root Cause Analysis: {self.target}",
            f"",
            f"Change: {self.baseline_mean:.2f} → {self.comparison_mean:.2f} ({direction} by {pct_change:.1f}%)",
            f"",
            f"Top Contributors:"
        ]
        
        sorted_contribs = sorted(
            self.contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for node, contrib in sorted_contribs[:5]:
            pct = abs(contrib / self.total_change * 100) if self.total_change != 0 else 0
            sign = "+" if contrib > 0 else ""
            lines.append(f"  {node}: {sign}{contrib:.2f} ({pct:.1f}%)")
        
        return "\n".join(lines)
    
    def top_contributors(self, n: int = 5) -> Dict[str, float]:
        sorted_items = sorted(
            self.contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return dict(sorted_items[:n])
    
    def plot(self, figsize=(10, 6)):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        sorted_contribs = sorted(
            self.contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        nodes = [x[0] for x in sorted_contribs]
        values = [x[1] for x in sorted_contribs]
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(nodes, values, color=colors)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Contribution to Change")
        ax.set_title(f"Root Causes of {self.target} Change")
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig, ax


@dataclass
class TreatmentEffectResult(BaseResult):
    
    treatment: str = ""
    outcome: str = ""
    ate: float = 0.0
    treated_mean: float = 0.0
    control_mean: float = 0.0
    n_treated: int = 0
    n_control: int = 0
    analysis_type: str = "treatment_effect"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "ate": self.ate,
            "treated_mean": self.treated_mean,
            "control_mean": self.control_mean,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "metadata": self.metadata
        }
    
    def summary(self) -> str:
        direction = "increases" if self.ate > 0 else "decreases"
        
        lines = [
            f"Treatment Effect Analysis",
            f"",
            f"Treatment: {self.treatment}",
            f"Outcome: {self.outcome}",
            f"",
            f"Average Treatment Effect (ATE): {self.ate:.4f}",
            f"",
            f"Interpretation: {self.treatment} {direction} {self.outcome} by {abs(self.ate):.4f} on average",
            f"",
            f"Treated group mean: {self.treated_mean:.4f} (n={self.n_treated})",
            f"Control group mean: {self.control_mean:.4f} (n={self.n_control})"
        ]
        
        return "\n".join(lines)


@dataclass
class WhatIfResult(BaseResult):
    
    interventions: Dict[str, Any] = field(default_factory=dict)
    samples: Optional[pl.DataFrame] = None
    baseline_means: Dict[str, float] = field(default_factory=dict)
    intervention_means: Dict[str, float] = field(default_factory=dict)
    analysis_type: str = "what_if"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "interventions": self.interventions,
            "baseline_means": self.baseline_means,
            "intervention_means": self.intervention_means,
            "n_samples": len(self.samples) if self.samples is not None else 0,
            "metadata": self.metadata
        }
    
    def summary(self) -> str:
        interventions_str = ", ".join(f"{k}={v}" for k, v in self.interventions.items())
        
        lines = [
            f"What-If Analysis",
            f"",
            f"Intervention: do({interventions_str})",
            f"Samples: {len(self.samples) if self.samples is not None else 0}",
            f"",
            f"Effects on outcomes:"
        ]
        
        for var in self.intervention_means:
            if var not in self.interventions:
                baseline = self.baseline_means.get(var, 0)
                new_val = self.intervention_means[var]
                diff = new_val - baseline
                sign = "+" if diff > 0 else ""
                lines.append(f"  {var}: {baseline:.4f} → {new_val:.4f} ({sign}{diff:.4f})")
        
        return "\n".join(lines)
    
    def effect_on(self, variable: str) -> Dict[str, float]:
        baseline = self.baseline_means.get(variable, 0)
        intervention = self.intervention_means.get(variable, 0)
        
        return {
            "baseline": baseline,
            "intervention": intervention,
            "difference": intervention - baseline,
            "relative_change": (intervention - baseline) / baseline if baseline != 0 else None
        }
    
    def mean(self, variable: str) -> float:
        if self.samples is None:
            raise ValueError("No samples available")
        return float(self.samples[variable].mean())