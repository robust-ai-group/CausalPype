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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    fig, ax = _make_fig(ax, figsize)
    G = model.graph
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")

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


def plot_effects(results, ax=None, figsize=(9, 5), title=None):
    """Forest plot of causal effect estimates, grouped by outcome variable.

    Y-axis shows 'treatment (ctrl → treat)' labels. When multiple outcome
    variables are present, results are grouped with dividers and outcome
    labels on the right margin.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if not isinstance(results, list):
        results = [results]

    scalar = [r for r in results if isinstance(r.estimate, (int, float, np.floating))]
    if not scalar:
        raise ValueError("No scalar estimates to plot.")

    # Group by outcome, preserve insertion order
    groups = defaultdict(list)
    for r in scalar:
        groups[r.details.get("outcome", "?")].append(r)

    multiple_outcomes = len(groups) > 1

    # Build y positions bottom-to-top so first outcome appears at top visually
    labels, values, y_pos = [], [], []
    group_spans    = []   # (y_lo, y_hi, outcome_name)
    group_dividers = []

    y = 0
    for i, (outcome, group_results) in enumerate(reversed(list(groups.items()))):
        if i > 0:
            group_dividers.append(y - 0.5)
            y += 1  # blank gap row between groups

        group_results = sorted(group_results, key=lambda r: abs(float(r.estimate)))
        y_lo = y
        for r in group_results:
            t  = r.details.get("treatment", "?")
            tv = r.details.get("treatment_value", "")
            cv = r.details.get("control_value", "")
            labels.append(f"{t}  ({cv} \u2192 {tv})")
            values.append(float(r.estimate))
            y_pos.append(y)
            y += 1
        group_spans.append((y_lo - 0.5, y - 0.5, outcome))

    auto_h = max(figsize[1], len(y_pos) * 0.65 + 1.5)
    fig, ax = _make_fig(ax, (figsize[0], auto_h))

    # Zero line
    ax.axvline(0, color=_PAL["dark"], lw=1.0, alpha=0.5, zorder=1)

    # Group dividers
    for yd in group_dividers:
        ax.axhline(yd, color=_PAL["border"], lw=0.8, alpha=0.25,
                   linestyle="--", zorder=1)

    # Stems + dots
    colors = [_PAL["green"] if v >= 0 else _PAL["red"] for v in values]
    for yi, vi, ci in zip(y_pos, values, colors):
        ax.plot([0, vi], [yi, yi], color=ci, lw=2.2, alpha=0.75,
                zorder=2, solid_capstyle="round")
        ax.scatter([vi], [yi], color=ci, s=80, zorder=3,
                   edgecolors="white", linewidths=1.2)

    # Value annotations
    max_abs = max(abs(v) for v in values) if values else 1
    pad = max_abs * 0.03
    for yi, vi, ci in zip(y_pos, values, colors):
        ha = "left" if vi >= 0 else "right"
        ax.text(vi + (pad if vi >= 0 else -pad), yi, f"{vi:+.3f}",
                va="center", ha=ha, fontsize=8.5, fontweight="bold", color=ci)

    # Outcome labels on right margin using blended axes/data transform
    if multiple_outcomes:
        for y_lo, y_hi, outcome in group_spans:
            ax.text(1.02, (y_lo + y_hi) / 2, outcome,
                    transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.5,
                    color=_PAL["blue"], fontweight="bold", fontstyle="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(min(y_pos) - 0.8, max(y_pos) + 0.8)

    if multiple_outcomes:
        ax.set_xlabel("Causal Effect  (do-calculus estimate)", fontsize=10)
    else:
        ax.set_xlabel(f"Effect on {list(groups.keys())[0]}", fontsize=10)

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


def plot_influences(result, ax=None, figsize=(9, 5), title=None):
    """Horizontal bar chart of intrinsic causal influences (normalised)."""
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


def plot_arrow_strength(result, ax=None, figsize=(9, 5), title=None, normalize=False):
    """Horizontal bar chart of arrow strengths (KL divergence).

    Parameters
    ----------
    normalize : bool
        If True, clip negatives to 0 and normalise so bars sum to 1,
        showing each parent's *share* of the total direct causal effect.
    """
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    strengths = result.details["strengths"]
    # Clip Monte Carlo negatives to 0 (finite-sample estimation noise)
    vals = {k: max(0.0, v) for k, v in strengths.items()}
    if normalize:
        total = sum(vals.values()) or 1.0
        vals = {k: v / total for k, v in vals.items()}

    sorted_items = sorted(vals.items(), key=lambda x: x[1])
    if not sorted_items:
        return fig, ax
    names, values = zip(*sorted_items)

    max_val = max(values)
    colors = [_PAL["red"] if v == max_val else _PAL["blue"] for v in values]
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, height=0.65, edgecolor="white", linewidth=0.5)

    for i, v in enumerate(values):
        lbl = f"{v:.1%}" if normalize else f"{v:.4f}"
        ax.text(v + max_val * 0.015, i, lbl, va="center", ha="left",
                fontsize=8.5, color=_PAL["dark"])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    xlabel = "Share of Direct Effect" if normalize else "Arrow Strength (KL Divergence)"
    ax.set_xlabel(xlabel, fontsize=10)
    target = result.details.get("target", "target")
    ax.set_title(title or f"Direct Causal Drivers of '{target}'",
                 fontsize=13, fontweight="bold", color=_PAL["dark"])
    _setup_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max_val * 1.2)
    fig.tight_layout()
    return fig, ax


def plot_anomalies(result, ax=None, figsize=(9, 5), title=None):
    """Horizontal bar chart of mean anomaly attribution scores."""
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    attrs = result.details.get("mean_attributions", {})
    if not attrs:
        raise ValueError("No anomaly attribution data found.")

    sorted_items = sorted(attrs.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in sorted_items][::-1]
    values = [v for _, v in sorted_items][::-1]
    colors = [_PAL["red"] if v > 0 else _PAL["blue"] for v in values]

    y = np.arange(len(names))
    ax.barh(y, values, color=colors, height=0.65, edgecolor="white", linewidth=0.5, alpha=0.85)

    max_abs = max(abs(v) for v in values) if values else 1
    for i, v in enumerate(values):
        ha = "left" if v >= 0 else "right"
        off = max_abs * 0.015 * (1 if v >= 0 else -1)
        ax.text(v + off, i, f"{v:+.4f}", va="center", ha=ha,
                fontsize=8.5, color=_PAL["dark"])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color=_PAL["dark"], lw=0.8, alpha=0.4)
    ax.set_xlabel("Mean Anomaly Attribution Score", fontsize=10)
    target = result.details.get("target", "target")
    ax.set_title(title or f"Anomaly Attribution: Root Causes of '{target}'",
                 fontsize=13, fontweight="bold", color=_PAL["dark"])
    _setup_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_distribution_change
# ---------------------------------------------------------------------------
def plot_distribution_change(result, ax=None, figsize=(9, 5), title=None):
    """Horizontal bars showing each node's contribution to a distribution shift.

    Positive (blue) = drives outcome *higher* in the new distribution;
    negative (red) = drives it *lower*.
    """
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    contributions = result.details.get("contributions", {})
    if not contributions:
        raise ValueError("No contributions data found in DistributionChange result.")

    # Drop near-zero nodes
    items = [(k, v) for k, v in contributions.items() if abs(v) > 1e-7]
    items = sorted(items, key=lambda x: abs(x[1]))          # ascending → top = largest

    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [_PAL["blue"] if v >= 0 else _PAL["red"] for v in values]
    y = np.arange(len(names))

    ax.barh(y, values, color=colors, height=0.65, edgecolor="white",
            linewidth=0.5, alpha=0.85)

    max_abs = max(abs(v) for v in values) if values else 1
    for i, v in enumerate(values):
        ha, off = ("left", max_abs * 0.015) if v >= 0 else ("right", -max_abs * 0.015)
        ax.text(v + off, i, f"{v:+.4f}", va="center", ha=ha,
                fontsize=8.5, color=_PAL["dark"])

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color=_PAL["dark"], lw=0.8, alpha=0.4)
    ax.set_xlabel("Contribution to Distribution Shift", fontsize=10)

    target = result.details.get("target", "target")
    ax.set_title(title or f"Distribution Change Attribution: '{target}'",
                 fontsize=13, fontweight="bold", color=_PAL["dark"])
    _setup_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig, ax


def plot_fairness_audit(result, ax=None, figsize=(7, 4), title=None):
    """Two-bar comparison: observational gap vs counterfactual disparity.

    Annotates the fold-reduction to make the mediation result immediately
    visible.
    """
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    d = result.details
    obs_gap = d.get("observational_gap") or 0.0
    cf_disp = d["counterfactual_disparity"]

    labels = ["Observational\nGap", "Counterfactual\nDisparity"]
    values = [obs_gap, cf_disp]
    colors = [_PAL["light_blue"], _PAL["blue"]]
    x = np.arange(2)

    bars = ax.bar(x, values, color=colors, width=0.45,
                  edgecolor=_PAL["border"], linewidth=0.8)

    top = max(values) * 1.4 if max(values) > 0 else 0.1
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + top * 0.03,
                f"{val:+.4f}  ({val:.1%})",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=_PAL["dark"])

    # Fold-reduction annotation
    if abs(obs_gap) > 1e-6 and abs(cf_disp) > 1e-6:
        fold = abs(obs_gap / cf_disp)
        ax.annotate(
            f"↓ {fold:.0f}× reduction",
            xy=(1, cf_disp + top * 0.03),
            xytext=(0.5, (obs_gap + cf_disp) / 2 + top * 0.05),
            arrowprops=dict(arrowstyle="->", color=_PAL["gray"], lw=1.2),
            fontsize=9, color=_PAL["gray"], ha="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Gap in Outcome Probability", fontsize=10)
    ax.set_ylim(0, top)

    attr = d.get("protected_attribute", "attribute")
    outcome = d.get("outcome", "outcome")
    ax.set_title(title or f"Fairness Audit — '{attr}' on '{outcome}'",
                 fontsize=12, fontweight="bold", color=_PAL["dark"])
    _setup_ax(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return fig, ax


def plot_cate_distribution(result, ax=None, figsize=(9, 5), title=None,
                            data=None, covariate=None, covariate_label=None,
                            bins=40):
    """Distribution of individual CATE effects.

    If *data* and *covariate* are provided, plots CATE vs. that covariate
    as a scatter with a binned-mean trend line; otherwise draws a histogram.
    """
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    effects = result.details.get("individual_effects")
    if effects is None:
        raise ValueError("No individual_effects in CATE result.")
    effects = np.asarray(effects).ravel()
    mean_eff = float(np.mean(effects))

    if covariate is not None and data is not None:
        cov = np.asarray(data[covariate].values[: len(effects)])
        ax.scatter(cov, effects, alpha=0.25, s=8, color=_PAL["blue"],
                   edgecolors="none", zorder=2)
        # Binned mean trend
        try:
            from scipy.stats import binned_statistic
            bm, be, _ = binned_statistic(cov, effects, statistic="mean", bins=20)
            bc = 0.5 * (be[:-1] + be[1:])
            ax.plot(bc, bm, color=_PAL["red"], lw=2.2, zorder=5,
                    label=f"Binned mean")
        except Exception:
            pass
        ax.axhline(mean_eff, color=_PAL["green"], lw=1.5, linestyle="--",
                   label=f"Mean = {mean_eff:+.4f}")
        ax.set_xlabel(covariate_label or covariate, fontsize=10)
        ax.set_ylabel("Individual Treatment Effect", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
    else:
        ax.hist(effects, bins=bins, color=_PAL["light_blue"],
                edgecolor="white", linewidth=0.4, density=True, alpha=0.85)
        ax.axvline(0, color=_PAL["dark"], lw=0.8, linestyle="--", alpha=0.4)
        ax.axvline(mean_eff, color=_PAL["red"], lw=2.0,
                   label=f"Mean = {mean_eff:+.4f}")
        ax.set_xlabel("Individual Treatment Effect", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)

    t = result.details.get("treatment", "T")
    o = result.details.get("outcome", "Y")
    ax.set_title(title or f"Heterogeneous Effects: {t} → {o}",
                 fontsize=13, fontweight="bold", color=_PAL["dark"])
    _setup_ax(ax)
    fig.tight_layout()
    return fig, ax

def plot_sensitivity(result, ax=None, figsize=(9, 5), title=None):
    """Compare the original ATE with the refutation test effects.

    A robust model shows: placebo ≈ 0, subset ≈ original,
    random common cause ≈ original.
    """
    import matplotlib.pyplot as plt

    fig, ax = _make_fig(ax, figsize)

    d = result.details
    original = d.get("original_ate", 0.0)

    method_labels = {
        "placebo": "Placebo\n(permuted T)",
        "subset": "Data Subset\n(80% resample)",
        "random_common_cause": "Random\nConfounder",
    }
    x_pos, x_labels, means, stds = [], [], [], []
    for i, (key, label) in enumerate(method_labels.items()):
        if key not in d:
            continue
        x_pos.append(i)
        x_labels.append(label)
        means.append(d[key]["mean_effect"])
        stds.append(d[key].get("std_effect", 0.0))

    ax.axhline(original, color=_PAL["blue"], lw=2.0, linestyle="--",
               label=f"Original ATE = {original:+.4f}", zorder=5)
    ax.axhline(0, color=_PAL["dark"], lw=0.6, alpha=0.3)

    ax.errorbar(x_pos, means, yerr=stds, fmt="o", color=_PAL["red"],
                markersize=8, capsize=5, capthick=1.5, lw=1.5, zorder=6)

    for xi, m in zip(x_pos, means):
        ax.text(xi, m + max(stds + [abs(original) * 0.05]) * 1.1,
                f"{m:+.4f}", ha="center", va="bottom", fontsize=8.5,
                color=_PAL["dark"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Estimated Effect", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)

    treatment = d.get("treatment", "T")
    ax.set_title(title or f"Sensitivity Analysis: '{treatment}'",
                 fontsize=13, fontweight="bold", color=_PAL["dark"])
    _setup_ax(ax)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return fig, ax