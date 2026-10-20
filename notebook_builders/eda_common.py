"""Shared pieces for the 2_EDA notebook builders (plain style, see BT4222-Group-10-main/2_eda.ipynb)."""
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
SETUP = r'''
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl            # only for the 70 M-row candidate table; everything small is pandas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Project root = folder that contains 1_Data_Preparation/processed (run 1_Data_Preparation/clean_and_validate_data.ipynb first)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/BT4222")
except ImportError:
    ROOT = next(p for p in [Path.cwd(), Path.cwd().parent] if (p / "1_Data_Preparation" / "processed").exists())
PROCESSED = ROOT / "1_Data_Preparation" / "processed"
FIG_DIR = ROOT / "2_EDA" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
print("Project root:", ROOT)
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
        self.append(nbf.v4.new_code_cell(one_statement_per_line(s.strip("\n"))))
