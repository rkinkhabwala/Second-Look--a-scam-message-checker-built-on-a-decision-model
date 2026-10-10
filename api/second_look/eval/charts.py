"""PNG charts for an eval run: reliability diagram and accuracy-vs-coverage curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (light surface): one series hue, recessive chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
SERIES_LIGHT = "#9ec5f4"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def reliability_diagram(bins: list[dict], ece: float, title: str, path: Path) -> None:
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(6, 6.4), height_ratios=[3, 1], sharex=True, facecolor=SURFACE
    )
    _style(ax)
    _style(ax_n)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.97, 0.90, "perfect calibration", color=MUTED, fontsize=8.5, ha="right", rotation=0)
    pts = [b for b in bins if b["count"]]
    ax.plot([b["mean_predicted"] for b in pts], [b["observed_rate"] for b in pts],
            color=SERIES, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed scam rate")
    ax.set_title(f"{title}\nECE (10 bins) = {ece:.3f}", color=INK, fontsize=10, loc="left")

    filled = [b for b in bins if b["count"]]
    ax_n.bar([(b["lo"] + b["hi"]) / 2 for b in filled], [b["count"] for b in filled],
             width=(bins[0]["hi"] - bins[0]["lo"]) * 0.88, color=SERIES_LIGHT)
    ax_n.set_yscale("log")
    ax_n.set_ylim(0.8, max(b["count"] for b in filled) * 2)
    ax_n.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_n.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_n.set_ylabel("Messages (log)")
    ax_n.set_xlabel("Predicted P(scam)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def coverage_chart(curve: list[dict], acc80: float, acc90: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.plot([c["coverage"] for c in curve], [c["accuracy"] for c in curve], color=SERIES, linewidth=2,
            clip_on=False)
    for cov, acc in ((0.8, acc80), (0.9, acc90)):
        ax.plot([cov], [acc], marker="o", markersize=8, color=SERIES,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, clip_on=False)
    lo = min(c["accuracy"] for c in curve)
    # At least a 10-point range so a near-flat curve isn't exaggerated.
    ax.set_ylim(max(0.0, min(lo - 0.02, 0.9)), 1.0)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Coverage (share of messages answered, most confident first)")
    ax.set_ylabel("Accuracy on answered messages")
    ax.set_title(
        f"{title}\nAccuracy {acc80:.1%} at 80% coverage · {acc90:.1%} at 90% coverage (marked)",
        color=INK, fontsize=10, loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def reliability_grid(panels: list[tuple[str, list[dict], float]], title: str, path: Path) -> None:
    """Small multiples: one reliability diagram per engine, shared axes, no second y-scale."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, (name, bins, ece) in zip(axes[0], panels):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        pts = [b for b in bins if b["count"]]
        ax.plot([b["mean_predicted"] for b in pts], [b["observed_rate"] for b in pts], color=SERIES,
                linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}\nECE {ece:.3f}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Predicted P(scam)")
    axes[0][0].set_ylabel("Observed scam rate")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
