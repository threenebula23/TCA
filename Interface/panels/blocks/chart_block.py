"""Terminal chart widget from ``lorne-chart`` JSON fences."""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

from rich.text import Text

from textual.containers import Vertical
from textual.widgets import Static


def _accent_color() -> str:
    try:
        from Interface.ui_prefs import load_prefs
        from Interface.themes import get_theme
        prefs = load_prefs()
        theme = get_theme(str(prefs.get("theme", "Purple Dark")))
        return str(prefs.get("accent_color") or theme.get("accent") or "#8B5CF6")
    except Exception:
        return "#8B5CF6"


def render_lorne_chart(chart_data: Dict[str, Any]) -> str:
    """Render lorne-chart JSON to plotext ASCII string."""
    if not isinstance(chart_data, dict):
        return "▤ chart: invalid data"

    labels: List[str] = [str(x) for x in (chart_data.get("labels") or [])]
    series = chart_data.get("series") or []
    if not labels and series:
        first = series[0] if isinstance(series[0], dict) else {}
        labels = [str(i + 1) for i in range(len(first.get("values") or []))]

    values: List[float] = []
    for s in series:
        if not isinstance(s, dict):
            continue
        raw = s.get("values") or []
        if raw:
            try:
                values = [float(v) for v in raw]
            except (TypeError, ValueError):
                values = []
            break
    if not values and labels:
        values = [0.0] * len(labels)
    if not labels:
        return "▤ chart: no labels"

    chart_type = str(chart_data.get("type") or "bar").lower()
    title = str(chart_data.get("title") or "").strip()

    try:
        import plotext as plt
    except ImportError:
        lines = [f"{title}:"] if title else []
        for lbl, val in zip(labels, values):
            lines.append(f"  {lbl}: {val}")
        return "\n".join(lines) if lines else "▤ plotext not installed"

    plt.clf()
    plt.theme("dark")
    plt.plotsize(48, 10)
    if title:
        plt.title(title)

    if chart_type == "line":
        plt.plot(labels, values, marker="dot")
    elif chart_type == "scatter":
        xs = list(range(1, len(values) + 1)) if not labels else list(range(len(labels)))
        plt.scatter(xs[: len(values)], values)
    else:
        plt.bar(labels, values)

    buf = io.StringIO()
    plt.savefig(buf)
    out = buf.getvalue().strip()
    return out or "▤ chart: empty render"


class ChartBlock(Vertical):
    """Collapsible-style chart card for the chat stream."""

    DEFAULT_CSS = """
    ChartBlock {
        height: auto;
        margin: 0 0 1 0;
        padding: 1 1;
        background: #12121A;
        border: round #2D2D3D;
    }
    ChartBlock .chart-title {
        height: auto;
        text-style: bold;
        margin: 0 0 1 0;
    }
    ChartBlock .chart-body {
        height: auto;
        color: #E5E7EB;
    }
    """

    def __init__(self, chart_data: Dict[str, Any], **kwargs) -> None:
        super().__init__(**kwargs)
        self._chart_data = chart_data if isinstance(chart_data, dict) else {}

    def compose(self):
        accent = _accent_color()
        title = str(self._chart_data.get("title") or "График")
        body = render_lorne_chart(self._chart_data)
        yield Static(Text.assemble(("▥ ", accent), (title, f"bold {accent}")), classes="chart-title")
        yield Static(body, classes="chart-body")
