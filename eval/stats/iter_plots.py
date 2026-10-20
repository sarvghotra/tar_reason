"""Shared helpers for the eval/stats scripts: seed statistics and paper
figures of score vs. iteration (mean +- s.e. over seeds, one line per run,
gain over iter-1 labelled at every later iteration).

Figures are sized for the CVPR page and saved as PDF (vector, TrueType text,
for LaTeX) plus a 300 dpi PNG preview. Needs matplotlib (the ``tar`` env).
"""

import math
from pathlib import Path


def mean_sem(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var / n)


def complete_seeds(runs, iters, keys, label=None):
    """Seeds that have every key at every iteration.

    With ``label``, warns about the seeds left out, since a figure then rests
    on fewer seeds than the tables.
    """
    kept = {
        s: r
        for s, r in runs.items()
        if all(all(k in r.get(it, {}) for k in keys) for it in iters)
    }
    dropped = [s for s in runs if s not in kept]
    if label is not None and dropped:
        print(
            f"[warn] {label}: figures use {len(kept)}/{len(runs)} seeds; "
            f"{', '.join(dropped)} lack some of iterations {iters}"
        )
    return kept


# CVPR page geometry (inches)
TEXT_WIDTH = 6.875
COLUMN_WIDTH = 3.25

# categorical slots in fixed order (one per run); these three stay
# colorblind-separable pairwise, a 4th hue would not -- compare <= 3 runs
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_MARKERS = ["o", "s", "^"]
# darker steps of the same hues for small text (>= 4.5:1 on white)
SERIES_TEXT_COLORS = ["#1d5fae", "#b8481b", "#0e7a54"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Times-like text to match the CVPR body font
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.6,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "legend.frameon": False,
            # embed TrueType so the PDF text stays editable / searchable
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def summarize(runs, iters, key, scale=1.0):
    """Per-iteration (mean, sem) over the seeds that have ``key``."""
    means, sems = [], []
    for it in iters:
        mean, sem = mean_sem([r[it][key] * scale for r in runs.values()])
        means.append(mean)
        sems.append(0.0 if math.isnan(sem) else sem)
    return means, sems


def draw_curve(ax, iters, means, sems, color, marker, label=None):
    """Mean line with markers and a +-1 s.e. band."""
    lo = [m - s for m, s in zip(means, sems)]
    hi = [m + s for m, s in zip(means, sems)]
    ax.fill_between(iters, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(
        iters,
        means,
        color=color,
        marker=marker,
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=1.5,
        label=label,
        zorder=3,
        clip_on=False,
    )
    return {"iters": iters, "means": means, "lo": lo, "hi": hi}


def annotate_gains(ax, curves, compact=False, digits=1):
    """Label every curve's gain over its iter-1 at each later iteration.

    Labels go just above the curve's band, or just below it when that would
    hit another run's band or label at the same iteration. Must run after the
    figure layout is final, since the gaps are measured in points.
    """
    fontsize = 6 if compact else 6.5
    pad, height, marker_r = 2.0, fontsize * 1.15, 2.5
    to_pt = 72.0 / ax.figure.dpi
    ymin, ymax = (
        ax.transData.transform([(0, y) for y in ax.get_ylim()])[:, 1] * to_pt
    )

    def y_pt(x, y):
        return ax.transData.transform((x, y))[1] * to_pt

    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    xs = sorted({it for c in curves for it in c["iters"][1:]})
    for x in xs:
        # every run's band (widened to the marker) at this x is an obstacle
        here = []
        for c in curves:
            if x not in c["iters"]:
                continue
            k = c["iters"].index(x)
            m = y_pt(x, c["means"][k])
            band = (
                min(y_pt(x, c["lo"][k]), m - marker_r),
                max(y_pt(x, c["hi"][k]), m + marker_r),
            )
            here.append((c, k, m, band))
        obstacles = [band for _, _, _, band in here]
        # top curve first so it claims the space above
        for c, k, m, band in sorted(here, key=lambda h: -h[2]):
            if k == 0:
                continue
            above = (band[1] + pad, band[1] + pad + height)
            below = (band[0] - pad - height, band[0] - pad)

            def cost(slot):
                hit = sum(overlap(slot, o) for o in obstacles if o != band)
                return hit + overlap(slot, (-1e9, ymin)) + overlap(slot, (ymax, 1e9))

            slot, va, dy = (above, "bottom", band[1] - m + pad)
            if cost(above) > 0 and cost(below) < cost(above):
                slot, va, dy = (below, "top", band[0] - m - pad)
            obstacles.append(slot)

            # typographic minus, not a hyphen, for losses
            gain = f"{c['means'][k] - c['means'][0]:+.{digits}f}".replace("-", "\u2212")
            ax.annotate(
                # narrow panels drop the Delta so neighbouring labels clear
                gain if compact else rf"$\Delta$ {gain}",
                xy=(x, c["means"][k]),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=fontsize,
                color=c["text_color"],
                zorder=4,
            )


def style_iter_axis(ax, iters):
    ax.set_xticks(iters)
    ax.set_xlim(iters[0] - 0.25, iters[-1] + 0.25)
    ax.margins(y=0.18)


def plot_panels(data, keys, labels, out_path, width, height, ncols=None,
                titles=False, ylabel=None, compact=False, scale=1.0, digits=1):
    """Score vs. iteration, one panel per key, one line per run.

    data     -- {run label: (runs, iters)}, runs = {seed: {iter: {key: value}}}
    labels   -- {key: text}; the y-label of each panel, or its title when
                ``titles`` is set (``ylabel`` then labels the first column)
    out_path -- path without extension; writes .pdf (vector) and .png
    scale    -- multiplier applied to every value before plotting
    digits   -- decimals of the gain-over-iter-1 labels
    """
    if len(data) > len(SERIES_COLORS):
        raise SystemExit(f"At most {len(SERIES_COLORS)} runs per figure")
    plt = setup_matplotlib()
    ncols = ncols or len(keys)
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), layout="constrained", squeeze=False
    )
    axes = axes.ravel()
    multi = len(data) > 1
    all_iters = sorted({it for _, iters in data.values() for it in iters})
    panel_curves = []
    for ax, key in zip(axes, keys):
        curves = []
        for i, (label, (runs, iters)) in enumerate(data.items()):
            means, sems = summarize(runs, iters, key, scale)
            curve = draw_curve(
                ax, iters, means, sems, SERIES_COLORS[i], SERIES_MARKERS[i],
                label=label,
            )
            # with several runs the label colour says which curve it belongs to
            curve["text_color"] = SERIES_TEXT_COLORS[i] if multi else INK_SECONDARY
            curves.append(curve)
        panel_curves.append(curves)
        style_iter_axis(ax, all_iters)
        ax.set_xlabel("Iteration")
        if titles:
            ax.set_title(labels[key])
        else:
            ax.set_ylabel(labels[key])
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    if titles and ylabel:
        for ax in axes[::ncols]:
            ax.set_ylabel(ylabel)
    if multi:
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="outside upper center",
            ncols=len(names),
            handlelength=1.8,
            columnspacing=1.5,
        )
    # fix the layout, then place the gain labels against the final geometry
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax, curves in zip(axes, panel_curves):
        annotate_gains(ax, curves, compact=compact, digits=digits)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_path.with_name(f"{out_path.name}.{ext}"))
    plt.close(fig)
    print(f"[fig] {out_path}.{{pdf,png}}")
