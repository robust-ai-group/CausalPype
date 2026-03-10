import numpy as np


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
_PAL = {
    "blue":        "#2563EB",
    "light_blue":  "#DBEAFE",
    "red":         "#EF4444",
    "light_red":   "#FEE2E2",
    "green":       "#059669",
    "amber":       "#F59E0B",
    "light_amber": "#FEF3C7",
    "gray":        "#6B7280",
    "light_gray":  "#F3F4F6",
    "dark":        "#1F2937",
    "border":      "#374151",
}


def _check_viz_deps():
    try:
        import matplotlib
        import seaborn
    except ImportError:
        raise ImportError(
            "Visualization dependencies not installed. "
            "Run: pip install 'causalpype[viz]'"
        )


def _setup_ax(ax):
    """Apply clean, minimal axes styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(labelsize=9, width=0.6)


def _make_fig(ax, figsize):
    """Create figure/axes if not provided."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _wrap_label(text, width=16):
    """Wrap a long label at underscores for readability."""
    if len(text) <= width:
        return text
    mid = len(text) // 2
    # Find the underscore nearest the midpoint
    left = text.rfind("_", 0, mid + 4)
    right = text.find("_", max(0, mid - 4))
    if left == -1 and right == -1:
        return text
    if left == -1:
        brk = right
    elif right == -1:
        brk = left
    else:
        brk = left if (mid - left) <= (right - mid) else right
    return text[:brk] + "\n" + text[brk + 1:]


# ---------------------------------------------------------------------------
# Hierarchical DAG layout
# ---------------------------------------------------------------------------
def _hierarchical_pos(G):
    """Compute layered positions for a DAG (roots at top, leaves at bottom).

    Uses topological depth for layer assignment and a barycenter heuristic
    to reduce edge crossings.
    """
    import networkx as nx

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(G)

    # Longest-path layering
    depth = {n: 0 for n in G.nodes}
    for node in topo:
        for child in G.successors(node):
            depth[child] = max(depth[child], depth[node] + 1)

    max_d = max(depth.values()) if depth else 0

    # Group by layer
    layers = {}
    for node, d in depth.items():
        layers.setdefault(d, []).append(node)
    for d in layers:
        layers[d] = sorted(layers[d])

    max_width = max(len(v) for v in layers.values())

    # Initial positioning
    pos = {}
    for d in sorted(layers.keys()):
        nodes = layers[d]
        n = len(nodes)
        spacing = max_width / max(n, 1)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2) * spacing
            y = (max_d - d) * 1.8
            pos[node] = (x, y)

    # Barycenter crossing-reduction (two sweeps)
    for _ in range(2):
        for d in sorted(layers.keys()):
            if d == 0:
                continue
            nodes = layers[d]
            bary = {}
            for node in nodes:
                parents = list(G.predecessors(node))
                bary[node] = np.mean([pos[p][0] for p in parents]) if parents else pos[node][0]
            layers[d] = sorted(nodes, key=lambda n: bary[n])
            n = len(layers[d])
            spacing = max_width / max(n, 1)
            for i, node in enumerate(layers[d]):
                pos[node] = ((i - (n - 1) / 2) * spacing, pos[node][1])

    return pos


# ---------------------------------------------------------------------------
# plot_graph
# ---------------------------------------------------------------------------
def plot_graph(model, strengths=None, ax=None, figsize=(14, 10), title=None):
    """Render the causal DAG with a hierarchical layout.

    Parameters
    ----------
    model : CausalModel
        Fitted causal model.
    strengths : dict, optional
        Edge-strength dict from ``ArrowStrength`` (keys like ``"X -> Y"``).
        Edges are coloured and sized proportionally.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional

    Returns
    -------
    (fig, ax)
    """
    _check_viz_deps()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    fig, ax = _make_fig(ax, figsize)
    G = model.graph
    pos = _hierarchical_pos(G)

    # Classify nodes
    roots = {n for n in G.nodes if G.in_degree(n) == 0}
    leaves = {n for n in G.nodes if G.out_degree(n) == 0}

    node_colors = []
    for n in G.nodes:
        if n in leaves:
            node_colors.append(_PAL["light_red"])
        elif n in roots:
            node_colors.append(_PAL["light_blue"])
        else:
            node_colors.append(_PAL["light_amber"])

    # --- Edges -----------------------------------------------------------
    # Accept TaskResult or dict
    if strengths is not None and hasattr(strengths, 'estimate'):
        strengths = strengths.estimate
    if strengths:
        abs_vals = [abs(v) for v in strengths.values()]
        max_s = max(abs_vals) if abs_vals else 1
        min_s = min(abs_vals) if abs_vals else 0
        rng = max_s - min_s if max_s > min_s else 1

        for u, v in G.edges:
            key = f"{u} -> {v}"
            s = abs(strengths.get(key, 0))
            norm = max(0.0, min(1.0, (s - min_s) / rng)) if s > 0 else 0.0
            width = 1.0 + 3.5 * norm
            alpha = 0.15 + 0.85 * norm
            color = _PAL["red"] if norm > 0.4 else _PAL["gray"]

            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                width=width, edge_color=color, alpha=alpha,
                arrows=True, arrowsize=15, arrowstyle="-|>",
                connectionstyle="arc3,rad=0.08",
                min_source_margin=22, min_target_margin=22,
            )

        # Label only the top-5 strongest edges
        ranked = sorted(strengths.items(), key=lambda x: abs(x[1]), reverse=True)
        top_labels = {}
        for edge_str, val in ranked[:5]:
            parts = edge_str.split(" -> ")
            if len(parts) == 2 and (parts[0], parts[1]) in G.edges:
                top_labels[(parts[0], parts[1])] = f"{val:.1f}"
        if top_labels:
            nx.draw_networkx_edge_labels(
                G, pos, top_labels, ax=ax, font_size=7,
                font_color=_PAL["dark"],
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
            )
    else:
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=1.2, edge_color=_PAL["gray"],
            arrows=True, arrowsize=15, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            min_source_margin=22, min_target_margin=22,
        )

    # --- Nodes -----------------------------------------------------------
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors,
        node_size=2200, edgecolors=_PAL["border"], linewidths=1.2,
    )

    wrapped = {n: _wrap_label(n) for n in G.nodes}
    nx.draw_networkx_labels(
        G, pos, labels=wrapped, ax=ax, font_size=7.5,
        font_weight="bold", font_color=_PAL["dark"],
    )

    # --- Legend ----------------------------------------------------------
    legend_items = [
        mpatches.Patch(fc=_PAL["light_blue"], ec=_PAL["border"], lw=0.8,
                       label="Exogenous (root)"),
        mpatches.Patch(fc=_PAL["light_amber"], ec=_PAL["border"], lw=0.8,
                       label="Endogenous"),
        mpatches.Patch(fc=_PAL["light_red"], ec=_PAL["border"], lw=0.8,
                       label="Outcome (leaf)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor=_PAL["light_gray"])

    ax.set_title(title or "Causal Graph", fontsize=13, fontweight="bold",
                 color=_PAL["dark"], pad=15)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_effects
# ---------------------------------------------------------------------------
def plot_effects(results, ax=None, figsize=(9, 5), title=None):
    """Lollipop chart of causal effect estimates.

    Labels are auto-generated from the task details
    (treatment name, control → treatment values).
    """
    _check_viz_deps()
    import matplotlib.pyplot as plt

    if not isinstance(results, list):
        results = [results]

    fig, ax = _make_fig(ax, figsize)

    labels, values = [], []
    for r in results:
        if not isinstance(r.estimate, (int, float, np.floating)):
            continue
        t = r.details.get("treatment", "?")
        o = r.details.get("outcome", "?")
        tv = r.details.get("treatment_value", "")
        cv = r.details.get("control_value", "")
        labels.append(f"{t}\n({cv} \u2192 {tv})")
        values.append(float(r.estimate))

    # Sort by absolute magnitude
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    y = np.arange(len(labels))
    colors = [_PAL["green"] if v >= 0 else _PAL["red"] for v in values]

    # Lollipop: stems + dots
    ax.hlines(y, 0, values, color=colors, linewidth=2.2, alpha=0.7)
    ax.scatter(values, y, color=colors, s=90, zorder=5,
               edgecolors="white", linewidths=1.2)

    # Annotations
    pad = 0.03 * (max(abs(v) for v in values) if values else 1)
    for i, (v, yi) in enumerate(zip(values, y)):
        ha = "left" if v >= 0 else "right"
        ax.text(v + (pad if v >= 0 else -pad), yi, f"{v:+.2f}",
                va="center", ha=ha, fontsize=9, fontweight="bold",
                color=colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color=_PAL["dark"], lw=0.6, alpha=0.3)

    outcome = results[0].details.get("outcome", "Outcome") if results else "Outcome"
    ax.set_xlabel(f"Effect on {outcome}", fontsize=10)
    ax.set_title(title or "Causal Effect Estimates", fontsize=13,
                 fontweight="bold", color=_PAL["dark"])

    _setup_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_causal_effect_curve
# ---------------------------------------------------------------------------
def plot_causal_effect_curve(result, ax=None, figsize=(9, 5), title=None):
    """Dose-response curve with \u00b11 SD confidence band."""
    _check_viz_deps()
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    responses = result.details.get("responses", [])
    if not responses:
        raise ValueError("No causal effect curve data in result.")

    t_vals = [r["treatment_value"] for r in responses]
    y_vals = [r["expected_outcome"] for r in responses]
    y_std = [r["std"] for r in responses]

    # Line + markers
    ax.plot(t_vals, y_vals, color=_PAL["blue"], linewidth=2.5,
            marker="o", markersize=5, markerfacecolor="white",
            markeredgecolor=_PAL["blue"], markeredgewidth=1.5, zorder=5)

    # Confidence band
    y_lo = [y - s for y, s in zip(y_vals, y_std)]
    y_hi = [y + s for y, s in zip(y_vals, y_std)]
    ax.fill_between(t_vals, y_lo, y_hi, alpha=0.12, color=_PAL["blue"],
                    label="\u00b11 SD")

    treatment = result.details.get("treatment", "Treatment")
    outcome = result.details.get("outcome", "Outcome")
    ax.set_xlabel(f"do({treatment})", fontsize=10)
    ax.set_ylabel(f"E[{outcome}]", fontsize=10)
    ax.set_title(title or f"Causal Effect Curve: {treatment} \u2192 {outcome}",
                 fontsize=13, fontweight="bold", color=_PAL["dark"])

    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.legend(fontsize=9, framealpha=0.9)
    _setup_ax(ax)
    fig.tight_layout()
    return fig, ax



# ---------------------------------------------------------------------------
# plot_influences
# ---------------------------------------------------------------------------
def plot_influences(result, ax=None, figsize=(9, 5), title=None):
    """Horizontal bar chart of intrinsic causal influences (normalised)."""
    _check_viz_deps()
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    normalized = result.details.get("normalized",
                                     result.details.get("influences", {}))
    if not normalized:
        raise ValueError("No influence data found.")

    sorted_items = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Gradient: strongest → dark blue, weakest → light blue
    n = len(values)
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.55 * (n - i) / n) for i in range(n)]

    y = np.arange(n)
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.5,
            height=0.7)

    for i, v in enumerate(values):
        if v > 0.06:
            ax.text(v - 0.005, y[i], f"{v:.1%}", va="center", ha="right",
                    fontsize=9, fontweight="bold", color="white")
        else:
            ax.text(v + 0.005, y[i], f"{v:.1%}", va="center", ha="left",
                    fontsize=9, fontweight="bold", color=_PAL["dark"])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Proportion of Variance Explained", fontsize=10)

    target = result.details.get("target", "target")
    ax.set_title(title or f"Intrinsic Causal Influence on \u2018{target}\u2019",
                 fontsize=13, fontweight="bold", color=_PAL["dark"])

    _setup_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(values) * 1.12 if values else 1)
    fig.tight_layout()
    return fig, ax


def plot_anomalies(result, ax=None, figsize=(9, 5), title=None):
    _check_viz_deps()
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)
    
    attrs = result.details["mean_attributions"]
    attrs_sorted = dict(sorted(attrs.items(), key=lambda x: x[1], reverse=True))

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in attrs_sorted.values()]
    ax.barh(list(attrs_sorted.keys()), list(attrs_sorted.values()), color=colors)
    ax.set_xlabel("Mean Anomaly Attribution Score")
    ax.set_title(title or "Anomalies")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    return fig, ax

def plot_arrow_strength(result, ax=None, figsize=(9, 5), title=None):
    _check_viz_deps()
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)
    
    strengths = result.details["strengths"]
    strengths_sorted = dict(sorted(strengths.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(strengths_sorted.keys()), list(strengths_sorted.values()), color="#2c3e50")
    ax.set_xlabel("Arrow Strength (KL divergence)")
    ax.set_title(title or "Arrow Strength")
    plt.tight_layout()
    return fig, ax