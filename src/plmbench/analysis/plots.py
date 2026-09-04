"""UMAP jointplots, sorted into results/figures/<label>/<family>/.

The KDE-overlay guard comes from downstream_umap_esmc.py, the only one of the
four drivers that did not crash on a hue group too small to estimate a density
for. The other three called plot_joint(sns.kdeplot) unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display on a compute node
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

from ..io import ensure_dir      # noqa: E402

# Figure subdirectory per hue column, matching the existing results/figures tree.
LABEL_DIRS = {
    "tax_domain": "taxonomy",
    "tax_family": "taxonomy",
    "immune_response": "immune_response",
}


def figure_path(figures_dir, hue: str, family: str, stem: str) -> Path:
    """results/figures/<taxonomy|immune_response>/<family>/<stem>.png"""
    sub = LABEL_DIRS.get(hue, hue)
    return ensure_dir(Path(figures_dir) / sub / family) / f"{stem}.png"


def jointplot(d: pd.DataFrame, hue: str, out_png, figsize: int = 40,
              min_kde_points: int = 10) -> bool:
    """KDE-overlaid jointplot that degrades gracefully on tiny/degenerate groups."""
    d = d.dropna(subset=[hue, "X", "Y"])
    if len(d) < 3:
        print(f"  skip {Path(out_png).name}: only {len(d)} points.")
        return False

    sns.set_theme(rc={"figure.figsize": (figsize, figsize)})
    g = sns.jointplot(data=d, x="X", y="Y", hue=hue)

    group_sizes = d.groupby(hue).size()
    if (group_sizes >= min_kde_points).all() and len(d) >= min_kde_points:
        try:
            g.plot_joint(sns.kdeplot, color="r")
            g.plot_marginals(sns.rugplot, color="r", clip_on=True)
        except Exception as e:                    # singular covariance, etc.
            print(f"  KDE skipped for {Path(out_png).name}: {e}")

    plt.legend(bbox_to_anchor=(1.05, 1.2), loc=2, borderaxespad=0.)
    plt.savefig(out_png)
    plt.close("all")
    print(f"  saved {Path(out_png).relative_to(Path(out_png).parents[2])}")
    return True


def plot_embedding(d: pd.DataFrame, name: str, family: str, figures_dir,
                   domains: list[str], figsize: int = 40,
                   min_kde_points: int = 10) -> int:
    """The full figure set for one embedding: taxonomy and immune response."""
    n = 0
    opts = dict(figsize=figsize, min_kde_points=min_kde_points)

    n += jointplot(d, "tax_domain",
                   figure_path(figures_dir, "tax_domain", family,
                               f"plot_{name}_domains"), **opts)

    for dom in domains:
        n += jointplot(d[d["tax_domain"] == dom], "tax_family",
                       figure_path(figures_dir, "tax_family", family,
                                   f"plot_{name}_{dom.lower()}_family"), **opts)

    n += jointplot(d, "immune_response",
                   figure_path(figures_dir, "immune_response", family,
                               f"plot_{name}_immuneresponse"), **opts)

    for dom in domains:
        n += jointplot(d[d["tax_domain"] == dom], "immune_response",
                       figure_path(figures_dir, "immune_response", family,
                                   f"plot_{name}_{dom.lower()}_immuneresponse"), **opts)
    return n
