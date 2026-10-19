from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

SPLITS = ("train", "validation", "test")


VERSION = "newsrec-v1"


@dataclass(frozen=True)
class Settings:
    seed: int = 4222
    text_dim: int = 64
    max_features: int = 50000
    min_df: int = 2
    title_weight: float = 0.7
    title_tokens: int = 30
    entity_dim: int = 100
    history_cap: int = 50
    snapshot_order_verified: bool = False
    recent_count: int = 10
    recent_hours: int = 24
    interest_count: int = 3
    ctr_strength: float = 20.0
    half_life_min_exposures: int = 100
    negative_count: int = 4
    update_mode: str = "prequential"
    context_features: bool = False
    parquet_batch_rows: int = 100000
    output_batch_rows: int = 100000

    def __post_init__(self):
        for key in (
            "text_dim",
            "max_features",
            "min_df",
            "title_tokens",
            "entity_dim",
            "history_cap",
            "recent_count",
            "recent_hours",
            "interest_count",
            "half_life_min_exposures",
            "negative_count",
            "parquet_batch_rows",
            "output_batch_rows",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 <= self.title_weight <= 1 or self.ctr_strength <= 0:
            raise ValueError("Invalid fusion/smoothing weights")
        if self.update_mode not in ("prequential", "frozen"):
            raise ValueError("update_mode must be prequential or frozen")

    def representation_fingerprint(self):
        fields = (
            "seed",
            "text_dim",
            "max_features",
            "min_df",
            "title_weight",
            "title_tokens",
            "entity_dim",
        )
        values = {name: getattr(self, name) for name in fields}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


DEFAULT_SETTINGS = Settings()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def normalise(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def normalised_mean(matrix, rows):
    if not len(rows):
        return (np.zeros(matrix.shape[1], dtype=np.float32), 0)
    selected = matrix[np.asarray(rows, dtype=np.int32)]
    valid = np.linalg.norm(selected, axis=1) > 1e-12
    if valid.any():
        pooled = normalise(selected[valid].mean(axis=0))
    else:
        pooled = np.zeros(matrix.shape[1], np.float32)
    return (pooled, int(valid.sum()))


def manifest(root, cfg, sources, outputs, **extra):
    root = Path(root)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    return dict(
        version=VERSION,
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        config=asdict(cfg),
        config_hash=cfg.fingerprint(),
        representation_config_hash=cfg.representation_fingerprint(),
        git_revision=revision,
        python=platform.python_version(),
        libraries={
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "pyarrow", "polars", "scikit-learn")
        },
        implementation_hashes={
            f"src/newsrec/{path.name}": sha256(path)
            for path in Path(__file__).parent.glob("*.py")
        },
        sources={str(Path(p).relative_to(root)): sha256(p) for p in sources},
        outputs={str(Path(p).relative_to(root)): sha256(p) for p in outputs},
        output_bytes=sum(Path(p).stat().st_size for p in outputs),
        **extra,
    )


@contextmanager
def measure(root, stage, **details):
    import threading

    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        worker.join()
        path = Path(root) / "results" / "representation_cost.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(
            stage=stage,
            wall_seconds=time.perf_counter() - start,
            sampled_peak_process_rss_mb=peak[0] / 1000000.0,
            platform=platform.platform(),
            cpu=platform.processor() or platform.machine(),
            logical_cpus=psutil.cpu_count(),
            ram_gb=psutil.virtual_memory().total / 1000000000.0,
            **details,
        )
        pd.DataFrame([row]).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )


def upsert_report(root, filename, frame, stage):
    path = Path(root) / "results" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame["stage"] = stage
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old[old.stage != stage], frame], ignore_index=True)
    frame.to_csv(path, index=False)


def require_inputs(root):
    p = Path(root) / "1_Data_Preparation" / "processed"
    files = [p / "news_catalogue.parquet", p / "users.parquet"]
    files += [
        p / f"{kind}_{split}.parquet"
        for kind in ("impressions", "candidates")
        for split in SPLITS
    ]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Run 1_Data_Preparation/clean_and_validate_data.ipynb first. Missing:\n"
            + "\n".join(missing)
        )
    return p


def saved_stage_ready(root, settings, stage):
    """Reuse disk outputs only when settings, inputs and output hashes agree."""
    path = Path(root) / "3_Representations/artifacts" / f"{stage}_manifest.json"
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    key = "config_hash" if stage == "profiles" else "representation_config_hash"
    expected = (
        settings.fingerprint()
        if stage == "profiles"
        else settings.representation_fingerprint()
    )
    if meta[key] != expected:
        raise ValueError(
            "Saved settings differ. Set REBUILD=True and rerun dependent notebooks."
        )
    for kind in ("sources", "outputs"):
        for relative, expected_hash in meta[kind].items():
            file = Path(root) / relative
            if not file.exists() or sha256(file) != expected_hash:
                raise ValueError(
                    f"Saved {stage} artifacts are stale or incomplete: {relative}. Set REBUILD=True."
                )
    return True


def prepare_feature_pairs(
    requests, candidate_pairs, state, bundle, cfg, candidate_source
):
    if state.cfg.fingerprint() != cfg.fingerprint():
        raise ValueError("State and feature configuration differ")
    if candidate_source not in ("logged", "retrieved"):
        raise ValueError("candidate_source must be logged or retrieved")
    if (
        requests.impression_uid.duplicated().any()
        or candidate_pairs.duplicated(["impression_uid", "news_code"]).any()
    ):
        raise ValueError("Duplicate request or candidate key")
    if not len(requests) or requests.timestamp.nunique() != 1:
        raise ValueError("Feature call requires one nonempty timestamp batch")
    t = pd.Timestamp(requests.timestamp.iloc[0])
    if state.active_time != t:
        raise ValueError("Call state.begin for this timestamp first")
    if not set(candidate_pairs.impression_uid) <= set(requests.impression_uid):
        raise ValueError("Candidate has no request")
    if not len(candidate_pairs):
        raise ValueError("Empty candidate batch")
    pairs = candidate_pairs[["impression_uid", "news_code"]].merge(
        requests[["impression_uid", "user_code"]],
        on="impression_uid",
        validate="many_to_one",
        sort=False,
    )
    rows = bundle.rows(pairs.news_code)
    eligible = bundle.eligible(pairs.news_code, t)
    if not eligible.all():
        raise ValueError("Candidate catalogue contains future/unavailable articles")
    return pairs, rows, t


def validate_feature_schema(schema):
    """Check a caller-defined model contract without owning its feature catalogue."""
    names = schema["model_columns"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one feature; feature names must be unique")
    protected = (
        set(schema["keys"]) | set(schema["excluded"]) | set(schema["diagnostics"])
    )
    if set(names) & protected:
        raise ValueError(
            f"Keys/targets/diagnostics cannot be model features: {sorted(set(names) & protected)}"
        )
    if names != [f["name"] for f in schema["features"]]:
        raise ValueError("Feature definitions and model_columns differ")
    return schema
