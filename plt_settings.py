base_width = 0.5
tick_major_base_ratio = 1.0
tick_minor_base_ratio = 0.5
tick_size_width_ratio = 3.0
tick_major_size_min = 3.0
tick_minor_size_min = 2.0

tick_major_width = tick_major_base_ratio * base_width
tick_minor_width = tick_minor_base_ratio * base_width
tick_major_size = max(tick_major_size_min, tick_size_width_ratio * tick_major_width)
tick_minor_size = max(tick_minor_size_min, tick_size_width_ratio * tick_minor_width)
plt_settings = {
    "figure.constrained_layout.use": True,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "font.size": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.titlesize": 9,
    "text.usetex": True,
    "text.latex.preamble": r"\renewcommand{\rmdefault}{ptm}\renewcommand{\sfdefault}{phv}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "axes.linewidth": base_width,
    "lines.linewidth": 2.0 * base_width,
    "xtick.major.width": tick_major_width,
    "ytick.major.width": tick_major_width,
    "xtick.minor.width": tick_minor_width,
    "ytick.minor.width": tick_minor_width,
    "xtick.major.size": tick_major_size,
    "ytick.major.size": tick_major_size,
    "xtick.minor.size": tick_minor_size,
    "ytick.minor.size": tick_minor_size,
    "grid.linewidth": base_width,
    "grid.linestyle": "solid",
    "grid.alpha": 0.2,
    "patch.linewidth": base_width,
    "axes.axisbelow": True,
    "text.color": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "black",
    "axes.facecolor": "none",
    "xtick.direction": "inout",
    "ytick.direction": "inout",
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.bottom": True,
    "legend.edgecolor": "inherit",
    "legend.facecolor": "w",
    "legend.shadow": False,
    "legend.frameon": True,
    "legend.fancybox": False,
}
