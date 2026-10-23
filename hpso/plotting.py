import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# impedance colour scale used in the paper (low: green/yellow -> red -> cyan/blue -> purple: high)
IMP_CMAP = LinearSegmentedColormap.from_list(
    "imp", ["#33ff33", "#ffff00", "#ff8800", "#ff0000", "#aa5500", "#44aa66", "#00ffff", "#0000ff", "#cc00cc"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("  saved", path)


def section(ax, data, t, cdp, cmap="seismic", vmin=None, vmax=None, title="", cbar_label="", well_cdp=None):
    if vmin is None and cmap == "seismic":
        vmax = np.percentile(np.abs(data), 99)
        vmin = -vmax
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[cdp[0] - 0.5, cdp[-1] + 0.5, t[-1], t[0]], interpolation="bilinear")
    ax.set_xlabel("CDP")
    ax.set_ylabel("Two-way time (ms)")
    ax.set_title(title)
    if well_cdp is not None:
        ax.axvline(well_cdp, color="k", lw=0.8, ls="--")
    cb = plt.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(cbar_label)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    return im
