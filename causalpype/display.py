from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if TYPE_CHECKING:
    from .tasks.base import TaskResult

# ── Type-based rendering rules ───────────────────────────────
#
# scalar (bool, int, float, str)  →  formatted value
# list/tuple of scalars            →  comma-separated
# list of dicts (uniform keys)     →  auto-table
# dict[str, number]                →  ranked bar chart
# dict with mixed scalar values    →  nested key-value section
# everything else                  →  skipped
#

_SCALAR = (bool, int, float, str, np.integer, np.floating, type(None))
_ACRONYMS = {"ate", "att", "atc", "nde", "nie", "ite", "cate", "sd", "std"}


def _label(key: str) -> str:
    """Convert snake_case key to Title Case label, preserving acronyms."""
    words = key.split("_")
    return " ".join(w.upper() if w.lower() in _ACRONYMS else w.title() for w in words)


def _arrow(value: float) -> str:
    if value > 0:
        return "\u2191"
    elif value < 0:
        return "\u2193"
    return " "


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return "\u2713" if value else "\u2717"
    if isinstance(value, float):
        return f"{_arrow(value)} {value:.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if value is None:
        return "-"
    return str(value)


def _is_scalar(value) -> bool:
    return isinstance(value, _SCALAR)


def _is_number_dict(d: dict) -> bool:
    """Dict where all values are int or float."""
    return bool(d) and all(isinstance(v, (int, float)) for v in d.values())


def _is_record_list(value) -> bool:
    """List of dicts with uniform keys (i.e. tabular data)."""
    if not isinstance(value, list) or len(value) == 0:
        return False
    if not all(isinstance(item, dict) for item in value):
        return False
    keys = set(value[0].keys())
    return all(set(item.keys()) == keys for item in value)


def _is_displayable(value) -> bool:
    """Can this value be rendered in some form?"""
    if _is_scalar(value):
        return True
    if isinstance(value, (list, tuple)) and all(_is_scalar(x) for x in value):
        return True
    if _is_record_list(value):
        return True
    if isinstance(value, dict):
        return True
    return False


# ── API ───────────────────────────────────────────────

def format_result(result: TaskResult) -> str:
    if HAS_RICH:
        return _format_rich(result)
    return _format_plain(result)


# ── Rich rendering ───────────────────────────────────────────

def _render_number_dict(d: dict) -> Table:
    """Ranked bar chart for dict[str, number]."""
    items = sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)
    max_val = max((abs(v) for _, v in items), default=1)
    bar_width = 20

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan", min_width=20)
    table.add_column(justify="right", min_width=10)
    table.add_column(min_width=bar_width)

    for key, val in items:
        filled = int(abs(val) / max_val * bar_width) if max_val > 0 else 0
        bar = "\u2588" * filled
        table.add_row(str(key), _format_scalar(val), bar)

    return table


def _render_record_list(records: list) -> Table:
    """Auto-table from list of dicts with uniform keys."""
    keys = list(records[0].keys())
    table = Table(show_header=True, box=None, padding=(0, 1))
    for k in keys:
        table.add_column(_label(k), justify="right" if isinstance(records[0].get(k), (int, float)) else "left")

    # Truncate long lists: show first/last 5
    if len(records) > 12:
        show = records[:5] + [None] + records[-5:]
    else:
        show = records

    for rec in show:
        if rec is None:
            table.add_row(*["..." for _ in keys])
        else:
            table.add_row(*[_format_scalar(rec.get(k)) for k in keys])

    return table


def _render_nested_dict(d: dict) -> Table:
    """Key-value table for dicts with mixed displayable values."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim", min_width=20)
    table.add_column()

    for k, v in d.items():
        if _is_scalar(v):
            table.add_row(_label(k), _format_scalar(v))
        elif isinstance(v, (list, tuple)) and all(_is_scalar(x) for x in v):
            table.add_row(_label(k), ", ".join(_format_scalar(x) for x in v))

    return table


def _render_detail(key: str, value, renderables: list):
    """Append renderable(s) for a single detail entry. Purely type-driven."""
    if _is_scalar(value):
        return "kv", _label(key), _format_scalar(value)

    if isinstance(value, (list, tuple)) and all(_is_scalar(x) for x in value):
        return "kv", _label(key), ", ".join(_format_scalar(x) for x in value)

    if _is_record_list(value):
        renderables.append(Text(""))
        renderables.append(Text(_label(key), style="bold"))
        renderables.append(_render_record_list(value))
        return None

    if isinstance(value, dict):
        if _is_number_dict(value):
            renderables.append(Text(""))
            renderables.append(Text(_label(key), style="bold"))
            renderables.append(_render_number_dict(value))
            return None
        # Mixed dict — render displayable scalars as nested kv
        displayable = {k: v for k, v in value.items() if _is_scalar(v) or (isinstance(v, (list, tuple)) and all(_is_scalar(x) for x in v))}
        if displayable:
            renderables.append(Text(""))
            renderables.append(Text(_label(key), style="bold"))
            renderables.append(_render_nested_dict(value))
        # Check for nested dicts-of-dicts (render as table if uniform)
        nested_records = {k: v for k, v in value.items() if isinstance(v, dict) and v}
        if nested_records:
            # Try to render as table if inner dicts share keys
            inner_keys = None
            uniform = True
            for v in nested_records.values():
                ks = set(v.keys())
                if inner_keys is None:
                    inner_keys = ks
                elif ks != inner_keys:
                    uniform = False
                    break
            if uniform and inner_keys:
                records = [{"": k, **v} for k, v in nested_records.items()]
                renderables.append(Text(""))
                renderables.append(_render_record_list(records))
        return None

    # Non-displayable type — skip
    return None


def _format_rich(result: TaskResult) -> str:
    renderables = []

    # Estimate
    est = result.estimate
    if isinstance(est, float):
        renderables.append(Text(f"Estimate: {_arrow(est)} {est:.4f}", style="bold"))
    elif isinstance(est, dict) and _is_number_dict(est):
        renderables.append(Text("Estimate:", style="bold"))
        renderables.append(_render_number_dict(est))
    elif isinstance(est, str):
        renderables.append(Text(f"Result: {est.upper().replace('_', ' ')}", style="bold"))
    else:
        renderables.append(Text(f"Estimate: {est}", style="bold"))

    # Details — type-driven
    kv_table = Table(show_header=False, box=None, padding=(0, 1))
    kv_table.add_column(style="dim", min_width=28)
    kv_table.add_column()
    has_kv = False
    est_is_dict = isinstance(est, dict)

    for key, value in result.details.items():
        # Skip values that duplicate the estimate
        if est_is_dict and isinstance(value, dict) and value == est:
            continue

        result_type = _render_detail(key, value, renderables)
        if result_type is not None:
            _, label, formatted = result_type
            kv_table.add_row(label, formatted)
            has_kv = True

    if has_kv:
        renderables.insert(1, kv_table)

    group = Group(*renderables)
    panel = Panel(group, title=f"[bold]{result.task_name}[/bold]", expand=False, padding=(1, 2))

    buf = StringIO()
    console = Console(file=buf, width=90, force_terminal=False)
    console.print(panel)
    return buf.getvalue().rstrip()


# ── Plain text fallback ──────────────────────────────────────

def _format_plain(result: TaskResult) -> str:
    title = result.task_name
    lines = [title, "\u2500" * len(title)]

    est = result.estimate
    if isinstance(est, float):
        lines.append(f"  Estimate: {_arrow(est)} {est:.4f}")
    elif isinstance(est, dict):
        lines.append("  Estimate:")
        for k, v in est.items():
            lines.append(f"    {k}: {_format_scalar(v)}")
    else:
        lines.append(f"  Estimate: {est}")

    est_is_dict = isinstance(est, dict)

    for key, value in result.details.items():
        if est_is_dict and isinstance(value, dict) and value == est:
            continue
        if not _is_displayable(value):
            continue

        label = _label(key)

        if _is_scalar(value):
            lines.append(f"  {label}: {_format_scalar(value)}")
        elif isinstance(value, (list, tuple)) and all(_is_scalar(x) for x in value):
            lines.append(f"  {label}: {', '.join(_format_scalar(x) for x in value)}")
        elif _is_record_list(value):
            lines.append(f"  {label}:")
            keys = list(value[0].keys())
            for rec in value[:10]:
                row = "  |  ".join(f"{k}={_format_scalar(rec[k])}" for k in keys)
                lines.append(f"    {row}")
            if len(value) > 10:
                lines.append(f"    ... ({len(value)} total)")
        elif isinstance(value, dict):
            if _is_number_dict(value):
                lines.append(f"  {label}:")
                for k, v in sorted(value.items(), key=lambda x: abs(x[1]), reverse=True):
                    lines.append(f"    {k:<35s} {_format_scalar(v)}")
            else:
                lines.append(f"  {label}:")
                for k, v in value.items():
                    if _is_scalar(v):
                        lines.append(f"    {_label(k)}: {_format_scalar(v)}")

    return "\n".join(lines)
