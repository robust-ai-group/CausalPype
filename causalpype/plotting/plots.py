from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx


def plot_causal_graph(
    graph,
    figsize: Tuple[int, int] = (12, 8),
    node_color: str = "#3498db",
    edge_color: str = "#7f8c8d",
    highlight_nodes: Optional[List[str]] = None,
    highlight_color: str = "#e74c3c",
    layout: str = "hierarchical",
    title: str = "Causal Graph",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    
    G = graph.networkx
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == "hierarchical":
        try:
            pos = _hierarchical_layout(G)
        except:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node colors
    node_colors = []
    highlight_nodes = highlight_nodes or []
    for node in G.nodes():
        if node in highlight_nodes:
            node_colors.append(highlight_color)
        else:
            node_colors.append(node_color)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_color,
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        alpha=0.7,
        width=1.5
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2000,
        alpha=0.9
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=9,
        font_weight='bold',
        font_color='white'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Legend for highlighted nodes
    if highlight_nodes:
        patches = [
            mpatches.Patch(color=node_color, label='Variables'),
            mpatches.Patch(color=highlight_color, label='Highlighted')
        ]
        ax.legend(handles=patches, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def _hierarchical_layout(G) -> Dict:
    # Get topological generations
    generations = list(nx.topological_generations(G))
    
    pos = {}
    for gen_idx, generation in enumerate(generations):
        n_nodes = len(generation)
        for node_idx, node in enumerate(sorted(generation)):
            x = (node_idx - (n_nodes - 1) / 2) * 1.5
            y = -gen_idx * 1.5
            pos[node] = (x, y)
    
    return pos


def plot_waterfall(
    result,
    figsize: Tuple[int, int] = (10, 6),
    positive_color: str = "#2ecc71",
    negative_color: str = "#e74c3c",
    total_color: str = "#3498db",
    max_items: int = 15,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    contributions = result.contributions
    
    # Sort by absolute contribution
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Limit items
    if len(sorted_items) > max_items:
        sorted_items = sorted_items[:max_items]
    
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate cumulative positions
    cumulative = 0
    bar_starts = []
    bar_values = []
    colors = []
    
    for val in values:
        bar_starts.append(cumulative if val >= 0 else cumulative + val)
        bar_values.append(abs(val))
        colors.append(positive_color if val >= 0 else negative_color)
        cumulative += val
    
    # Add total bar
    labels.append("TOTAL")
    bar_starts.append(0 if result.total_change >= 0 else result.total_change)
    bar_values.append(abs(result.total_change))
    colors.append(total_color)
    
    # Create horizontal bars
    y_pos = np.arange(len(labels))
    
    ax.barh(y_pos, bar_values, left=bar_starts, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, (start, val, orig_val) in enumerate(zip(bar_starts, bar_values, values + [result.total_change])):
        x_pos = start + val / 2
        ax.text(x_pos, i, f"{orig_val:+.3f}", ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
    
    # Add connecting lines
    cumulative = 0
    for i, val in enumerate(values):
        cumulative += val
        if i < len(values) - 1:
            ax.plot([cumulative, cumulative], [i + 0.4, i + 0.6], 
                   color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Contribution to Change")
    
    title = title or f"Root Cause Analysis: {result.target}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add subtitle with baseline/comparison info
    subtitle = f"Baseline: {result.baseline_mean:.3f} → Comparison: {result.comparison_mean:.3f}"
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', fontsize=10, color='gray')
    
    # Legend
    patches = [
        mpatches.Patch(color=positive_color, label='Positive'),
        mpatches.Patch(color=negative_color, label='Negative'),
        mpatches.Patch(color=total_color, label='Total')
    ]
    ax.legend(handles=patches, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_treatment_effect(
    result,
    figsize: Tuple[int, int] = (8, 5),
    control_color: str = "#3498db",
    treated_color: str = "#e74c3c",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    fig, ax = plt.subplots(figsize=figsize)
    
    x = [0, 1]
    y = [result.control_mean, result.treated_mean]
    colors = [control_color, treated_color]
    labels = ["Control", "Treated"]
    
    bars = ax.bar(x, y, color=colors, width=0.5, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * max(y),
                f"{val:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add ATE annotation
    ate_text = f"ATE = {result.ate:+.4f}"
    ax.annotate(
        ate_text,
        xy=(0.5, max(y) * 0.5),
        fontsize=12,
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(result.outcome, fontsize=11)
    
    title = title or f"Treatment Effect: {result.treatment} → {result.outcome}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_distribution_comparison(
    baseline_data: np.ndarray,
    intervention_data: np.ndarray,
    variable_name: str,
    figsize: Tuple[int, int] = (10, 5),
    baseline_color: str = "#3498db",
    intervention_color: str = "#e74c3c",
    baseline_label: str = "Baseline",
    intervention_label: str = "After Intervention",
    kind: str = "both",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    fig, ax = plt.subplots(figsize=figsize)
    
    if kind in ["hist", "both"]:
        # Determine bins
        all_data = np.concatenate([baseline_data, intervention_data])
        bins = np.linspace(all_data.min(), all_data.max(), 30)
        
        ax.hist(baseline_data, bins=bins, alpha=0.5, color=baseline_color, 
                label=baseline_label, density=True, edgecolor='white')
        ax.hist(intervention_data, bins=bins, alpha=0.5, color=intervention_color,
                label=intervention_label, density=True, edgecolor='white')
    
    if kind in ["kde", "both"]:
        from scipy import stats
        
        x_range = np.linspace(
            min(baseline_data.min(), intervention_data.min()),
            max(baseline_data.max(), intervention_data.max()),
            200
        )
        
        try:
            kde_baseline = stats.gaussian_kde(baseline_data)
            kde_intervention = stats.gaussian_kde(intervention_data)
            
            ax.plot(x_range, kde_baseline(x_range), color=baseline_color, 
                   linewidth=2, label=f"{baseline_label} (KDE)" if kind == "both" else baseline_label)
            ax.plot(x_range, kde_intervention(x_range), color=intervention_color,
                   linewidth=2, label=f"{intervention_label} (KDE)" if kind == "both" else intervention_label)
        except:
            pass  # KDE can fail with small samples
    
    # Add mean lines
    ax.axvline(baseline_data.mean(), color=baseline_color, linestyle='--', 
               linewidth=2, alpha=0.8)
    ax.axvline(intervention_data.mean(), color=intervention_color, linestyle='--',
               linewidth=2, alpha=0.8)
    
    # Add mean annotations
    ymax = ax.get_ylim()[1]
    ax.annotate(f"μ={baseline_data.mean():.2f}", 
                xy=(baseline_data.mean(), ymax * 0.9),
                color=baseline_color, fontsize=10, fontweight='bold')
    ax.annotate(f"μ={intervention_data.mean():.2f}",
                xy=(intervention_data.mean(), ymax * 0.8),
                color=intervention_color, fontsize=10, fontweight='bold')
    
    ax.set_xlabel(variable_name, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    
    title = title or f"Distribution Comparison: {variable_name}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_forest(
    effects: Dict[str, Tuple[float, float, float]],
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#3498db",
    title: str = "Treatment Effects",
    xlabel: str = "Effect Size",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(effects.keys())
    values = [effects[name][0] for name in names]
    ci_lower = [effects[name][1] for name in names]
    ci_upper = [effects[name][2] for name in names]
    
    y_pos = np.arange(len(names))
    
    # Error bars
    xerr_lower = [v - l for v, l in zip(values, ci_lower)]
    xerr_upper = [u - v for v, u in zip(values, ci_upper)]
    
    ax.errorbar(values, y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='o', color=color, capsize=5, capthick=2,
                markersize=8, linewidth=2)
    
    # Add zero line
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    # Add value labels
    for i, (val, lower, upper) in enumerate(zip(values, ci_lower, ci_upper)):
        ax.text(val, i + 0.3, f"{val:.3f} [{lower:.3f}, {upper:.3f}]",
                ha='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_what_if_comparison(
    result,
    variables: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    baseline_color: str = "#3498db",
    intervention_color: str = "#e74c3c",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    # Get variables to plot (exclude intervened variables)
    if variables is None:
        variables = [v for v in result.intervention_means.keys() 
                    if v not in result.interventions]
    
    # Filter to variables with both baseline and intervention values
    variables = [v for v in variables 
                if v in result.baseline_means and v in result.intervention_means]
    
    if not variables:
        raise ValueError("No variables to plot")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(variables))
    width = 0.35
    
    baseline_vals = [result.baseline_means[v] for v in variables]
    intervention_vals = [result.intervention_means[v] for v in variables]
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color=baseline_color, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, intervention_vals, width, label='After Intervention',
                   color=intervention_color, edgecolor='white', linewidth=1)
    
    # Add change annotations
    for i, (b, a) in enumerate(zip(baseline_vals, intervention_vals)):
        change = a - b
        pct_change = (change / b * 100) if b != 0 else 0
        y_pos = max(b, a) + 0.02 * max(baseline_vals + intervention_vals)
        ax.text(i, y_pos, f"{change:+.2f}\n({pct_change:+.1f}%)",
                ha='center', fontsize=8, color='gray')
    
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.set_ylabel("Value")
    
    # Title with interventions info
    interventions_str = ", ".join([f"{k}={v}" for k, v in result.interventions.items()])
    title = title or f"What-If Analysis"
    ax.set_title(f"{title}\nInterventions: {interventions_str}", fontsize=11, fontweight='bold')
    
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_tornado(
    sensitivities: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6),
    positive_color: str = "#2ecc71",
    negative_color: str = "#e74c3c",
    title: str = "Sensitivity Analysis",
    xlabel: str = "Effect on Outcome",
    max_items: int = 15,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    # Sort by absolute value
    sorted_items = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    
    if len(sorted_items) > max_items:
        sorted_items = sorted_items[:max_items]
    
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [positive_color if v >= 0 else negative_color for v in values]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(names))
    
    ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, val in enumerate(values):
        x_pos = val + 0.01 * max(abs(v) for v in values) * np.sign(val)
        ax.text(x_pos, i, f"{val:.3f}", ha='left' if val >= 0 else 'right',
                va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig