"""Import the editable ranking methods from this trusted project checkout.

Only tagged definitions and feature selections are loaded, never notebook setup,
examples, outputs or dataset builds. Restart Python after editing the notebook.
This keeps one editable feature implementation visible to notebook reviewers.
"""

import ast
import hashlib
from collections import deque
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from .common import (
    DEFAULT_SETTINGS,
    normalise,
    prepare_feature_pairs,
    validate_feature_schema,
)
from .temporal import HOUR, User
from .temporal import TemporalState as _StateStorage


def load_ranking_methods():
    """Return an independent namespace of the notebook's current definitions.

    The notebook is executable source code from the local trusted repository;
    do not replace it with an untrusted downloaded notebook.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "4_Feature_Engineering/4_01_build_ranking_features.ipynb"
    )
    namespace = {
        "__name__": __name__,
        "np": np,
        "pd": pd,
        "hashlib": hashlib,
        "deque": deque,
        "DEFAULT_SETTINGS": DEFAULT_SETTINGS,
        "normalise": normalise,
        "prepare_feature_pairs": prepare_feature_pairs,
        "validate_feature_schema": validate_feature_schema,
        "_StateStorage": _StateStorage,
        "User": User,
        "HOUR": HOUR,
    }
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tags = set(cell.metadata.get("tags", []))
        if tags & {"feature_catalogue", "feature_controls"}:
            tree = ast.parse(cell.source)
        elif tags & {"method", "feature_schema"}:
            tree = ast.parse(cell.source)
            tree.body = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            ]
        else:
            continue
        exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - trusted repository source
    return namespace


_methods = load_ranking_methods()
CausalProfiles = _methods["CausalProfiles"]
TemporalState = _methods["TemporalState"]
build_features = _methods["build_features"]
feature_schema = _methods["feature_schema"]
sample_training_groups = _methods["sample_training_groups"]
