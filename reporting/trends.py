"""Per-patient trend charts: one small panel per measurement over time."""
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from records.catalog import entries  # noqa: E402

NORMAL_BAND = "#10b981"
LINE = "#4f46e5"
MAX_PANELS = 9


def trend_png(modality, series, dpi=110):
    """PNG bytes, or None when no measurement has a value to plot.

    series: {key: [(study_date, value_or_None, study_id), ...]} oldest first.
    Unmeasured points are gaps, never zeros. The adult reference range is
    shaded so a drift out of range is visible at a glance.
    """
    panels = []
    for key, label, unit, rng in entries(modality):
        points = [(datetime.strptime(d, "%Y-%m-%d"), v)
                  for d, v, _sid in series.get(key, []) if v is not None]
        if points:
            panels.append((label, unit, rng, points))
    if not panels:
        return None
    panels = panels[:MAX_PANELS]

    cols = 3 if len(panels) > 4 else min(2, len(panels))
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows),
                             squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    # The grid can have spare cells (switched off above), so not strict.
    for ax, (label, unit, rng, points) in zip(axes.ravel(), panels, strict=False):
        xs, ys = zip(*points, strict=True)
        if rng is not None:
            lo, hi = rng
            ax.axhspan(lo if lo is not None else min(min(ys), hi or 0) - 1e6,
                       hi if hi is not None else max(max(ys), lo or 0) + 1e6,
                       color=NORMAL_BAND, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=LINE, marker="o", lw=1.6, ms=5)
        ax.set_title(f"{label}" + (f" ({unit})" if unit else ""), fontsize=10,
                     fontweight="bold")
        pad = (max(ys) - min(ys)) * 0.25 or abs(max(ys)) * 0.1 or 1.0
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
        ax.tick_params(labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
