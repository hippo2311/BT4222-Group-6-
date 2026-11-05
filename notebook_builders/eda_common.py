"""Shared pieces for the 2_EDA notebook builders. Style follows BT4222-Group-10/2_eda.ipynb: pandas only,
print() summaries, numbered step comments, markdown with findings and model implications."""
import re
import nbformat as nbf
from pathlib import Path

HEADER_RE = re.compile(r"^(\s*)((?:for|if|elif|else|while|with|try|except|finally|def)\b[^;]*?:)\s*(\S.*)$")

def _split_statements(line):
    """Split one source line at semicolons that sit outside strings and brackets."""
    parts, depth, quote, start = [], 0, None, 0
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "#" and depth == 0:
            break
        elif ch == ";" and depth == 0:
            parts.append(line[start:i]); start = i + 1
        i += 1
    parts.append(line[start:])
    return [p for p in parts if p.strip()]


def one_statement_per_line(src):
    out = []
    for line in src.splitlines():
        parts = _split_statements(line)
        if len(parts) <= 1:
            out.append(line)
            continue
        indent = re.match(r"\s*", line).group(0)
        m = HEADER_RE.match(parts[0])
        if m:                      # "for x in y: a; b" -> header line, then indented a and b
            out.append(m.group(1) + m.group(2))
            out.append(indent + "    " + m.group(3).strip())
            out.extend(indent + "    " + p.strip() for p in parts[1:])
        else:
            out.extend(indent + p.strip() for p in parts)
    return "\n".join(out)

EDA_DIR = Path(__file__).resolve().parent.parent / "2_EDA"

# One short setup cell: paths, imports, a plot style. No framework.
SETUP = r"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Project root = the folder that contains 1_Data_Preparation/processed (written by 1. data_preparation.ipynb)
CODE_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "run_all.py").is_file())
ROOT = Path(os.environ.get("NEWSREC_ROOT", CODE_ROOT)).expanduser().resolve()
PROCESSED = ROOT / "1_Data_Preparation" / "processed"
FIG_DIR = ROOT / "2_EDA" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
print("Project root:", ROOT)
"""


def write_notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    out = EDA_DIR / f"{name}.ipynb"
    nbf.write(nb, out)
    print("wrote", out, len(cells), "cells")


PLOT_RE = re.compile(r"^# \d+\. Plots?\b:?\s*(.*)$")


def split_plots(src):
    """Split a section's source into one computation cell and one cell per figure.

    The computation part ends at the first "# N. Plot" step comment; the plotting part is then cut at blank lines,
    so each figure (plt.figure / plt.subplots block) becomes its own cell. Variables carry over between cells."""
    lines = src.splitlines()
    idx = next((i for i, line in enumerate(lines) if PLOT_RE.match(line)), None)
    if idx is None:
        return [src]
    cells = ["\n".join(lines[:idx]).rstrip()]
    suffix = PLOT_RE.match(lines[idx]).group(1).strip()
    chunks, cur = [], []
    for line in lines[idx + 1:]:
        if line.strip():
            cur.append(line)
        elif cur:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    for k, chunk in enumerate(chunks):
        label = "# Plot" if len(chunks) == 1 else f"# Plot {k + 1} of {len(chunks)}"
        if k == 0 and suffix:
            label += ": " + suffix
        cells.append("\n".join([label] + chunk))
    return cells


class Cells(list):
    def md(self, s):
        self.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(self, s):
        for part in split_plots(one_statement_per_line(s.strip("\n"))):
            self.append(nbf.v4.new_code_cell(part))
