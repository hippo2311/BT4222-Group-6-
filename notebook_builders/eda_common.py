"""Shared pieces for the three 2_EDA notebook builders."""
import nbformat as nbf
from pathlib import Path

EDA_DIR = Path(__file__).resolve().parent.parent / "2_EDA"

SETUP_PATHS = r'''
from pathlib import Path
import sys

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def has_processed(root):
    return (root / "1_Data_Preparation" / "processed" / "candidates_train.parquet").is_file()

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    candidate_roots = [Path("/content/drive/MyDrive/BT4222"), Path("/content/drive/MyDrive/MIND"), Path("/content/drive/MyDrive"), Path("/content")]
else:
    here = Path.cwd()
    candidate_roots = [here, here.parent, here.parent.parent, Path.home() / "BT4222"]
ROOT = next((r for r in candidate_roots if has_processed(r)), None)
if ROOT is None:
    raise FileNotFoundError("Run 1_Data_Preparation/clean_and_validate_data.ipynb first; processed files not found under "
                            + ", ".join(map(str, candidate_roots)))
PROCESSED = ROOT / "1_Data_Preparation" / "processed"
EDA_DIR = ROOT / "2_EDA"
FIG_DIR, TABLE_DIR = EDA_DIR / "figures", EDA_DIR / "tables"
for d in (FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("Environment:", "Google Colab" if IN_COLAB else "local", "| project root:", ROOT)
print("figures ->", FIG_DIR, "| tables ->", TABLE_DIR)
'''

SETUP_LIBS = r'''
import importlib, subprocess
def ensure(module, pip_name=None):
    try:
        importlib.import_module(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or module])
for _m in ["polars", "pandas", "numpy", "matplotlib", "scipy"]:
    ensure(_m)

import json, time, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats
from IPython.display import display, Markdown
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 180); pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
SPLIT = "train"                       # all descriptive figures use the training split (team decision, 1 Oct 2026)

# ---- one visual system for every figure: categorical slots in fixed order, text in ink, grid recessive, one y-axis per panel
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#e4e4e1", "#8c8b86"
DIVERGING = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f2ef", BLUE])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c9c5",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 8.5, "figure.facecolor": "white",
})
PCT = FuncFormatter(lambda v, _: f"{v:g}%")
THOUSANDS = FuncFormatter(lambda v, _: f"{v/1e6:g}M" if abs(v) >= 1e6 else (f"{v/1e3:g}k" if abs(v) >= 1e3 else f"{v:g}"))

manifest = {"notebook": NOTEBOOK_NAME, "run_stamp": RUN_STAMP, "split": SPLIT, "polars": pl.__version__, "tables": [], "figures": [], "headline": {}}
H = manifest["headline"]
tables = {}

def save_table(name, df):
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    tables[name] = df
    if path.name not in manifest["tables"]:
        manifest["tables"].append(path.name)
    return df

def save_fig(fig, name, caption=None):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    if path.name not in manifest["figures"]:
        manifest["figures"].append(path.name)
    plt.show()
    if caption:
        display(Markdown(f"*{name}.png — {caption}*"))

def pct(a, b):
    return 100.0 * a / b if b else float("nan")

def wilson(clicks, n, z=1.96):
    clicks, n = np.asarray(clicks, float), np.asarray(n, float)
    p = clicks / n; denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half

def describe(series, name):
    s = series.drop_nulls()
    q = lambda p: float(s.quantile(p, "nearest"))
    return {"statistic": name, "count": s.len(), "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()),
            "p5": q(0.05), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "p99": q(0.99), "max": float(s.max())}

def bucket(col, edges, labels):
    """Right-closed bins: value <= edges[0] -> labels[0], ..., value > edges[-1] -> labels[-1]."""
    expr = pl.when(pl.col(col) <= edges[0]).then(pl.lit(labels[0]))
    for e, l in zip(edges[1:], labels[1:]):
        expr = expr.when(pl.col(col) <= e).then(pl.lit(l))
    return expr.otherwise(pl.lit(labels[-1]))

def ordered(df, col, order):
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col).reset_index(drop=True)

print("Setup ok ·", RUN_STAMP, "· polars", pl.__version__)
'''

CLOSING = r'''
manifest["tables"] = sorted(set(manifest["tables"])); manifest["figures"] = sorted(set(manifest["figures"]))
(EDA_DIR / f"manifest_{NOTEBOOK_NAME}.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"{len(manifest['figures'])} figures in {FIG_DIR}\n{len(manifest['tables'])} tables in {TABLE_DIR}\nheadline numbers -> manifest_{NOTEBOOK_NAME}.json")
'''


def write_notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    out = EDA_DIR / f"{name}.ipynb"
    nbf.write(nb, out)
    print("wrote", out, len(cells), "cells")


class Cells(list):
    def md(self, s):
        self.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(self, s):
        self.append(nbf.v4.new_code_cell(s.strip("\n")))

    def setup(self, notebook_name):
        self.code(SETUP_PATHS)
        self.code(f'NOTEBOOK_NAME = "{notebook_name}"\n' + SETUP_LIBS)
