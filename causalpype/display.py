"""Rich, task-aware display formatting for TaskResult objects.

Usage::

    result = cp.ATE("X", "Y").run(model)
    print(result)          # rich formatted output
    result.summary()       # same thing, returns str

This module is the text equivalent of ``plotting.py``: it takes TaskResult
objects and renders them as human-readable strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tasks.base import TaskResult

# ── Utility functions ─────────────────────────────────────────

def _arrow(value: float) -> str:
    if value > 0:
        return "\u2191"  # ↑
    elif value < 0:
        return "\u2193"  # ↓
    return " "


def _check(passed) -> str:
    return "\u2713" if passed else "\u2717"  # ✓ / ✗


def _bar(value: float, max_value: float, width: int = 25) -> str:
    if max_value <= 0:
        return ""
    filled = int(abs(value) / max_value * width)
    return "\u2588" * filled  # █


def _header(title: str) -> str:
    return f"{title}\n\u2500" * 0 + f"{title}\n{'\u2500' * len(title)}"


def _kv(key: str, value, kw: int = 28, decimals: int = 2) -> str:
    if isinstance(value, float):
        v = f"{value:.{decimals}f}"
    elif isinstance(value, (list, tuple)):
        v = ", ".join(str(x) for x in value)
    else:
        v = str(value)
    return f"  {key:<{kw}s} {v}"


def _fmt(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _sorted_dict(d: dict, reverse: bool = True) -> list:
    """Sort dict items by absolute value, descending."""
    return sorted(d.items(), key=lambda x: abs(x[1]), reverse=reverse)


# ── Registry ──────────────────────────────────────────────────

_FORMATTERS: dict[str, callable] = {}


def _register(task_name: str):
    def decorator(fn):
        _FORMATTERS[task_name] = fn
        return fn
    return decorator


def format_result(result: TaskResult) -> str:
    """Format a TaskResult into rich, human-readable text."""
    formatter = _FORMATTERS.get(result.task_name, _format_generic)
    return formatter(result)


# ── Generic fallback ──────────────────────────────────────────

_SKIP = {
    "samples", "individual_effects", "predictions", "residuals",
    "anomalies", "estimator", "cate_model", "policy",
    "individual_counterfactuals", "all_ite", "ite_treated",
    "ite_control", "response_df", "assignments",
    "counterfactual_samples", "interventional_samples",
    "noise_data", "observed_data", "raw_attributions", "raw_strengths",
    "individual_nde", "individual_nie", "lower_bound", "upper_bound",
}


def _format_generic(result: TaskResult) -> str:
    lines = [_header(result.task_name)]
    est = result.estimate
    if isinstance(est, float):
        lines.append(f"  Estimate: {_fmt(est)}")
    elif isinstance(est, dict):
        lines.append("  Estimate:")
        for k, v in est.items():
            lines.append(f"    {k}: {_fmt(v) if isinstance(v, float) else v}")
    else:
        lines.append(f"  Estimate: {est}")
    for k, v in result.details.items():
        if k in _SKIP:
            continue
        if isinstance(v, float):
            lines.append(f"  {k}: {_fmt(v)}")
        elif isinstance(v, dict):
            lines.append(f"  {k}:")
            for dk, dv in v.items():
                lines.append(f"    {dk}: {_fmt(dv) if isinstance(dv, float) else dv}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ── Task-specific formatters ──────────────────────────────────

@_register("ATE")
def _format_ate(r: TaskResult) -> str:
    d = r.details
    treat = d.get("treatment", "?")
    outcome = d.get("outcome", "?")
    t_val = d.get("treatment_value", "?")
    c_val = d.get("control_value", "?")
    est = r.estimate
    lines = [
        _header(f"ATE: {treat} \u2192 {outcome}"),
        _kv("Contrast:", f"{c_val} \u2192 {t_val}"),
        _kv("Estimate:", f"{_arrow(est)} {_fmt(est)}"),
        _kv("Samples:", d.get("num_samples", "?")),
    ]
    return "\n".join(lines)


@_register("Arrow Strength")
def _format_arrow_strength(r: TaskResult) -> str:
    d = r.details
    target = d.get("target", "?")
    strengths = r.estimate if isinstance(r.estimate, dict) else d.get("strengths", {})
    items = _sorted_dict(strengths)
    max_val = max((abs(v) for _, v in items), default=1)

    lines = [_header(f"Arrow Strength \u2192 {target}")]
    for edge, val in items:
        bar = _bar(val, max_val)
        lines.append(f"  {edge:<38s} {val:>8.2f}  {bar}")
    return "\n".join(lines)


@_register("Intrinsic Causal Influence")
def _format_ici(r: TaskResult) -> str:
    d = r.details
    target = d.get("target", "?")
    norm = d.get("normalized", {})
    items = _sorted_dict(norm)
    max_val = max((abs(v) for _, v in items), default=1)

    lines = [_header(f"Intrinsic Causal Influence \u2192 {target}")]
    for node, frac in items:
        bar = _bar(frac, max_val)
        lines.append(f"  {node:<38s} {frac:>6.1%}  {bar}")
    total = d.get("total_variance_explained")
    if total is not None:
        lines.append(f"\n  Total variance explained: {_fmt(total)}")
    return "\n".join(lines)


@_register("Mediation")
def _format_mediation(r: TaskResult) -> str:
    d = r.details
    treat = d.get("treatment", "?")
    outcome = d.get("outcome", "?")
    t_val = d.get("treatment_value", "?")
    c_val = d.get("control_value", "?")
    mediators = d.get("mediators", [])

    te = d.get("total_effect", 0)
    nde = d.get("natural_direct_effect", 0)
    nie = d.get("natural_indirect_effect", 0)
    prop = d.get("proportion_mediated")

    lines = [
        _header(f"Mediation: {treat} \u2192 {outcome}"),
        _kv("Contrast:", f"{c_val} \u2192 {t_val}"),
        _kv("Mediators:", ", ".join(str(m) for m in mediators) if mediators else "auto-detected"),
        "",
        _kv("Total Effect:", f"{_arrow(te)} {_fmt(te)}"),
        _kv("Natural Direct (NDE):", f"{_arrow(nde)} {_fmt(nde)}"),
        _kv("Natural Indirect (NIE):", f"{_arrow(nie)} {_fmt(nie)}"),
    ]
    if prop is not None and isinstance(prop, (int, float)) and not np.isnan(prop):
        lines.append(_kv("Proportion Mediated:", f"{prop:.1%}"))
    return "\n".join(lines)


@_register("Fairness Audit")
def _format_fairness(r: TaskResult) -> str:
    d = r.details
    attr = d.get("protected_attribute", "?")
    outcome = d.get("outcome", "?")
    priv = d.get("privileged_value", "?")
    unpriv = d.get("unprivileged_value", "?")

    disp = d.get("counterfactual_disparity", r.estimate)
    gap = d.get("observational_gap", 0)
    mean_unf = d.get("mean_individual_unfairness", 0)
    max_unf = d.get("max_individual_unfairness", 0)
    n_priv = d.get("n_privileged", "?")
    n_unpriv = d.get("n_unprivileged", "?")

    lines = [
        _header(f"Fairness Audit: {attr} \u2192 {outcome}"),
        _kv("Privileged:", f"n={n_priv:,}" if isinstance(n_priv, int) else str(n_priv)),
        _kv("Unprivileged:", f"n={n_unpriv:,}" if isinstance(n_unpriv, int) else str(n_unpriv)),
        "",
        _kv("Counterfactual Disparity:", f"{_arrow(disp)} {_fmt(disp)}"),
        _kv("Observational Gap:", f"{_arrow(gap)} {_fmt(gap)}"),
        _kv("Mean Individual Unfairness:", _fmt(mean_unf)),
        _kv("Max Individual Unfairness:", _fmt(max_unf)),
    ]
    return "\n".join(lines)


@_register("CATE")
def _format_cate(r: TaskResult) -> str:
    d = r.details
    treat = d.get("treatment", "?")
    outcome = d.get("outcome", "?")
    modifiers = d.get("effect_modifiers", [])
    method = d.get("method", "?")
    mean = d.get("mean_effect", r.estimate)
    std = d.get("std_effect", 0)
    bounds = d.get("bounds", (None, None))

    lines = [
        _header(f"CATE: {treat} \u2192 {outcome}"),
        _kv("Effect Modifiers:", ", ".join(str(m) for m in modifiers)),
        _kv("Method:", method),
        "",
        _kv("Mean Effect:", f"{_arrow(mean)} {_fmt(mean)}"),
        _kv("Std:", _fmt(std)),
    ]
    if bounds and bounds[0] is not None:
        lines.append(_kv("Range:", f"[{_fmt(bounds[0])}, {_fmt(bounds[1])}]"))
    return "\n".join(lines)


@_register("Dose-Response")
def _format_dose_response(r: TaskResult) -> str:
    d = r.details
    treat = d.get("treatment", "?")
    outcome = d.get("outcome", "?")
    responses = d.get("responses", [])

    lines = [_header(f"Dose-Response: {treat} \u2192 {outcome}")]

    if responses:
        n = len(responses)
        lines.append(f"  {n} points")
        lines.append("")

        # Show first 3 and last 3 if many, otherwise all
        show = responses if n <= 8 else responses[:3] + [None] + responses[-3:]
        for resp in show:
            if resp is None:
                lines.append("  ...")
                continue
            tv = resp.get("treatment_value", "?")
            ey = resp.get("expected_outcome", "?")
            sd = resp.get("std", None)
            tv_s = _fmt(tv) if isinstance(tv, float) else str(tv)
            ey_s = _fmt(ey) if isinstance(ey, float) else str(ey)
            sd_s = f"  (SD {_fmt(sd)})" if sd is not None and isinstance(sd, float) else ""
            lines.append(f"  do({treat}={tv_s}) \u2192 E[{outcome}] = {ey_s}{sd_s}")

        # Range summary
        outcomes = [resp["expected_outcome"] for resp in responses
                    if isinstance(resp.get("expected_outcome"), (int, float))]
        if outcomes:
            lines.append(f"\n  E[{outcome}] range: {_fmt(min(outcomes))} to {_fmt(max(outcomes))}")
    return "\n".join(lines)


@_register("Counterfactual")
def _format_counterfactual(r: TaskResult) -> str:
    d = r.details
    interventions = d.get("interventions", {})
    outcome = d.get("outcome")
    n_units = d.get("n_units", "?")
    interv_str = ", ".join(f"{k}={v}" for k, v in interventions.items()) if isinstance(interventions, dict) else str(interventions)

    lines = [
        _header("Counterfactual Analysis"),
        _kv("Interventions:", interv_str),
        _kv("N units:", n_units),
    ]

    if outcome:
        lines.append(_kv("Outcome:", outcome))
        fm = d.get("factual_mean")
        cm = d.get("counterfactual_mean")
        me = d.get("mean_effect")
        lines.append("")
        if fm is not None:
            lines.append(_kv("Factual Mean:", _fmt(fm)))
        if cm is not None:
            lines.append(_kv("Counterfactual Mean:", _fmt(cm)))
        if me is not None:
            lines.append(_kv("Average Effect:", f"{_arrow(me)} {_fmt(me)}"))
    else:
        # Dict estimate — show all node means
        est = r.estimate
        if isinstance(est, dict):
            lines.append("")
            lines.append("  Counterfactual Means:")
            for k, v in est.items():
                lines.append(f"    {k:<30s} {_fmt(v) if isinstance(v, float) else v}")
    return "\n".join(lines)


@_register("KNN Intervention")
def _format_knn(r: TaskResult) -> str:
    d = r.details
    treat = d.get("treatment", "?")
    outcome = d.get("outcome", "?")
    k = d.get("k", "?")
    match_on = d.get("match_on", [])
    t_val = d.get("treatment_value", 1)
    c_val = d.get("control_value", 0)

    ate = d.get("ate", r.estimate)
    att = d.get("att", 0)
    atc = d.get("atc", 0)
    n_t = d.get("n_treated", "?")
    n_c = d.get("n_control", "?")
    mq_t = d.get("match_quality_treated")
    mq_c = d.get("match_quality_control")

    lines = [
        _header(f"KNN Matching: {treat} \u2192 {outcome}"),
        _kv("Contrast:", f"{c_val} \u2192 {t_val}"),
        _kv("K:", k),
        _kv("Match on:", ", ".join(str(m) for m in match_on) if match_on else "all"),
        "",
        _kv("ATE:", f"{_arrow(ate)} {_fmt(ate)}"),
        _kv("ATT:", f"{_arrow(att)} {_fmt(att)}"),
        _kv("ATC:", f"{_arrow(atc)} {_fmt(atc)}"),
        _kv("N treated:", n_t),
        _kv("N control:", n_c),
    ]
    if mq_t is not None:
        lines.append(_kv("Match quality (treated):", _fmt(mq_t, 4)))
    if mq_c is not None:
        lines.append(_kv("Match quality (control):", _fmt(mq_c, 4)))
    return "\n".join(lines)


@_register("Intervention")
def _format_intervention(r: TaskResult) -> str:
    d = r.details
    interventions = d.get("interventions", {})
    outcome = d.get("outcome")
    interv_str = ", ".join(f"{k}={v}" for k, v in interventions.items()) if isinstance(interventions, dict) else str(interventions)

    lines = [
        _header("Intervention (do-calculus)"),
        f"  do({interv_str})",
    ]

    if outcome:
        mean = d.get("mean", r.estimate)
        std = d.get("std")
        lines.append(f"\n  E[{outcome}] = {_fmt(mean)}" +
                     (f"  (SD {_fmt(std)})" if std is not None else ""))
    else:
        est = r.estimate
        if isinstance(est, dict):
            lines.append("\n  Expected Values:")
            for k, v in est.items():
                lines.append(f"    {k:<30s} {_fmt(v) if isinstance(v, float) else v}")
    return "\n".join(lines)


@_register("Stochastic Intervention")
def _format_stochastic(r: TaskResult) -> str:
    d = r.details
    treat = d.get("treatment", "?")
    outcome = d.get("outcome", "?")
    shift = d.get("shift", "?")
    is_binary = d.get("is_binary", False)
    baseline = d.get("E[Y|baseline]")
    shifted = d.get("E[Y|shifted]")
    effect = r.estimate

    shift_desc = f"p += {shift}" if is_binary else f"{'+' if isinstance(shift, (int, float)) and shift >= 0 else ''}{shift}"

    lines = [
        _header(f"Stochastic Intervention: {treat} \u2192 {outcome}"),
        _kv("Shift:", f"{shift_desc} ({'binary' if is_binary else 'continuous'})"),
        "",
    ]
    if baseline is not None:
        lines.append(_kv("E[Y | baseline]:", _fmt(baseline)))
    if shifted is not None:
        lines.append(_kv("E[Y | shifted]:", _fmt(shifted)))
    if isinstance(effect, (int, float)):
        lines.append(_kv("Effect:", f"{_arrow(effect)} {_fmt(effect)}"))
    return "\n".join(lines)


@_register("Anomaly Attribution")
def _format_anomaly(r: TaskResult) -> str:
    d = r.details
    target = d.get("target", "?")
    n_anom = d.get("n_anomalies", 0)
    attribs = r.estimate if isinstance(r.estimate, dict) else d.get("mean_attributions", {})

    if not attribs:
        error = d.get("error", "No anomalies found")
        return f"{_header(f'Anomaly Attribution \u2192 {target}')}\n  {error}"

    items = _sorted_dict(attribs)
    max_val = max((abs(v) for _, v in items), default=1)

    lines = [
        _header(f"Anomaly Attribution \u2192 {target}"),
        f"  Anomalous cases: {n_anom}",
        "",
    ]
    for node, val in items:
        bar = _bar(val, max_val)
        lines.append(f"  {node:<38s} {_arrow(val)} {abs(val):.4f}  {bar}")
    return "\n".join(lines)


@_register("Distribution Change")
def _format_dist_change(r: TaskResult) -> str:
    d = r.details
    target = d.get("target", "?")
    n_old = d.get("n_old", "?")
    n_new = d.get("n_new", "?")
    contribs = r.estimate if isinstance(r.estimate, dict) else d.get("contributions", {})

    items = _sorted_dict(contribs)
    max_val = max((abs(v) for _, v in items), default=1)

    n_old_s = f"{n_old:,}" if isinstance(n_old, int) else str(n_old)
    n_new_s = f"{n_new:,}" if isinstance(n_new, int) else str(n_new)

    lines = [
        _header(f"Distribution Change \u2192 {target}"),
        f"  Old: n={n_old_s}  |  New: n={n_new_s}",
        "",
    ]
    for node, val in items:
        bar = _bar(val, max_val)
        lines.append(f"  {node:<38s} {_arrow(val)} {abs(val):.4f}  {bar}")
    return "\n".join(lines)


@_register("Validation")
def _format_validate(r: TaskResult) -> str:
    status = str(r.estimate).upper().replace("_", " ")
    lines = [_header(f"Model Validation: {status}")]

    d = r.details
    struct = d.get("structure")
    if struct:
        edge_tests = struct.get("edge_tests", {})
        n_pass = sum(1 for v in edge_tests.values() if v.get("success"))
        n_fail = len(edge_tests) - n_pass

        lines.append("")
        lines.append(f"  Edge tests: {n_pass} passed, {n_fail} failed")

        # Show passing first, then failing
        for edge, info in sorted(edge_tests.items()):
            passed = info.get("success")
            p = info.get("p_value")
            p_str = f"(p={p:.4f})" if isinstance(p, float) else ""
            lines.append(f"  {_check(passed)} {edge:<40s} {p_str}")

        # Local Markov tests
        node_details = struct.get("node_details", {})
        markov_lines = []
        for node, tests in node_details.items():
            markov = tests.get("local_markov", {})
            if markov:
                passed = markov.get("success")
                p = markov.get("p_value")
                p_str = f"(p={p:.4f})" if isinstance(p, float) else ""
                markov_lines.append(f"  {_check(passed)} {node:<40s} {p_str}")
        if markov_lines:
            lines.append("")
            lines.append("  Local Markov tests:")
            lines.extend(markov_lines)

    model_test = d.get("model")
    if model_test:
        result = model_test.get("result", "N/A")
        lines.append(f"\n  Model test (invertibility): {result}")

    return "\n".join(lines)
