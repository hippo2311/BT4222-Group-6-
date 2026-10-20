"""Builds 1_Data_Preparation/clean_and_validate_data.ipynb (nbformat v4)."""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "1_Data_Preparation" / "clean_and_validate_data.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 1. Data cleaning and validation — MIND-large, team split

This notebook takes the three split folders produced by `0_Data_Acquisition/download_split_process.ipynb`
(`train/`, `validation/`, `test/`, each holding `behaviors_<split>.parquet` and `news_<split>.parquet`) and

1. re-derives the **split procedure** from the saved files and checks it (row counts, time windows, boundary ties, id collisions);
2. **parses** every history and impression list and checks their grammar, labels, duplicates and missing values;
3. checks the **links between behaviour and news records** (every referenced article exists, every news row is referenced);
4. audits **news-field quality**: blank text, empty versus missing entity lists, invalid JSON, duplicate titles, entity-embedding coverage;
5. runs **temporal-integrity checks** so that no future information can enter an earlier impression
   (static histories, ordered windows, user and article overlap between splits);
6. documents every **cleaning decision** and writes tidy, typed **processed files** that the EDA and modelling notebooks share;
7. writes `data_quality_report.csv` (one row per check: value, expectation, PASS / WARN / INFO / FAIL).

Nothing is deleted from the data: every impression in the three releases is kept. The "cleaning" consists of parsing,
typing, sorting, adding stable keys and recording the caveats that later notebooks must respect.

**Run order.** Run top to bottom (about 3 to 5 minutes locally, 16 GB RAM is enough). The notebook finds the split folders
next to its own parent folder (project root) or on Google Drive in Colab. Outputs go to `1_Data_Preparation/processed/`
and `1_Data_Preparation/data_quality_report.csv`.
""")

code(r"""
from pathlib import Path
import sys
import os
import json

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DATA_ROOT = None  # Normally leave this as None: use this project's local folders.
SPLITS = ['train', 'validation', 'test']
SPLIT_CODE = {split: i for i, split in enumerate(SPLITS)}
DRIVE_URL = 'https://drive.google.com/drive/folders/1rkP9G4Bxp1xjhFW1COErvms-CMeNtJUd'
REQUIRED_INPUTS = tuple(f'{split}/{kind}_{split}.parquet'
    for split in SPLITS for kind in ('behaviors', 'news')) + (
        'train/entity_embedding.vec', 'test/entity_embedding.vec')
RECOVERY_MESSAGE = (
    'Run step 0 first: 0_Data_Acquisition/download_split_process.ipynb '
    '(terminal: python run_all.py --from-stage 0 --to-stage 0), '
    'then rerun step 1. Step 0 uses original MIND releases or locally downloaded ZIPs, not the shared Drive copy.')

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
here = Path.cwd().resolve()
candidate_roots = [here, *here.parents, Path('/content/drive/MyDrive/BT4222')]
if DATA_ROOT is not None:
    ROOT = Path(DATA_ROOT).expanduser().resolve()
elif os.environ.get('NEWSREC_ROOT'):
    ROOT = Path(os.environ['NEWSREC_ROOT']).expanduser().resolve()
else:
    ROOT = next((p for p in candidate_roots if (p / 'run_all.py').is_file()),
                next((p for p in candidate_roots if (p / 'train/behaviors_train.parquet').is_file()),
                     here.parent if here.name == '1_Data_Preparation' else here))
print('Local input/output root:', ROOT)

""")

code(r"""
def input_ready(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix == '.parquet' or '.parquet.part' in path.name:
            import pyarrow.parquet as pq
            pq.read_metadata(path)
        else:
            with path.open('rb') as stream:
                first = stream.readline(65536).lower()
            if b'<html' in first or b'<!doctype' in first:
                return False
        return True
    except (OSError, ValueError):
        return False


def ensure_local_inputs(root, drive_url=DRIVE_URL, required_inputs=REQUIRED_INPUTS):
    # This function is inline: running this notebook never executes another notebook.
    root = Path(root).resolve()
    marker = root / '0_Data_Acquisition/split_manifest.json'
    if marker.exists():
        try:
            building = json.loads(marker.read_text()).get('status') == 'building'
        except ValueError as exc:
            raise RuntimeError('The step-0 build marker is unreadable. ' + RECOVERY_MESSAGE) from exc
        if building:
            raise RuntimeError('A step-0 split rebuild was interrupted. ' + RECOVERY_MESSAGE)
    missing = [relative for relative in required_inputs if not input_ready(root / relative)]
    if not missing:
        print('All required inputs are local. Google Drive is not contacted.')
        return {'downloaded': [], 'reused': list(required_inputs)}
    bad = [relative for relative in missing if (root / relative).exists()]
    if bad:
        raise RuntimeError('Local inputs are empty/unreadable: ' + ', '.join(bad) + '. ' + RECOVERY_MESSAGE)
    print('Missing local inputs; fetching the agreed splits/vectors from Google Drive:', missing)
    try:
        import gdown
        entries = gdown.download_folder(url=drive_url, output=str(root), skip_download=True,
            quiet=True, use_cookies=False, timeout=30)
        by_path = {}
        for entry in entries or []:
            # Only exact known input filenames are accepted; remote notebook/code files are ignored.
            relative = str(entry.path).replace('\\', '/')
            if relative in required_inputs:
                if relative in by_path:
                    raise ValueError(f'Duplicate Drive filename: {relative}')
                by_path[relative] = entry
        absent = [relative for relative in missing if relative not in by_path]
        if absent:
            raise FileNotFoundError('Required files missing from Drive: ' + ', '.join(absent))
        downloaded = []
        for relative in missing:
            target = root / relative
            if not target.resolve().is_relative_to(root):
                raise ValueError('Input path escapes the selected data root.')
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(target.name + '.part')
            if staging.exists() and not input_ready(staging):
                staging.unlink()
            print('DOWNLOAD', relative, flush=True)
            result = gdown.download(id=by_path[relative].id, output=str(staging), quiet=False,
                use_cookies=False, resume=True, retries=3, timeout=30)
            if result is None or not input_ready(staging):
                raise RuntimeError(f'Download incomplete: {relative}')
            if target.exists():
                raise FileExistsError(f'Destination appeared during download: {target}')
            staging.replace(target)
            downloaded.append(relative)
        remaining = [relative for relative in required_inputs if not input_ready(root / relative)]
        if remaining:
            raise FileNotFoundError('Inputs still missing: ' + ', '.join(remaining))
        print('Required inputs downloaded locally; continuing data preparation.')
        return {'downloaded': downloaded, 'reused': [p for p in required_inputs if p not in downloaded]}
    except Exception as exc:
        # Keep successful earlier downloads; a retry only requests remaining files.
        raise RuntimeError('Could not obtain the agreed dataset from Google Drive '
            '(permissions, quota, network, or missing files). Already-local files are kept. '
            + RECOVERY_MESSAGE) from exc

""")

code(r"""
input_status = ensure_local_inputs(ROOT)
NB_DIR = ROOT / '1_Data_Preparation'
PROCESSED = NB_DIR / 'processed'
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_PATH = NB_DIR / 'data_quality_report.csv'
print('Environment:', 'Google Colab' if IN_COLAB else 'local')
print('Processed files ->', PROCESSED)
print('Quality report ->', REPORT_PATH)

""")

code(r"""
import importlib, subprocess
def ensure(module, pip_name=None):
    try:
        importlib.import_module(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or module])
for _m in ["polars", "pandas", "numpy", "pyarrow"]:
    ensure(_m)

import hashlib, json, platform, re, time, warnings
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display, Markdown
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option("display.max_columns", 30); pd.set_option("display.width", 180); pd.set_option("display.max_colwidth", 120)

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"       # only used if a file still stores time as text
EXPECTED_TRAIN_LOG_ROWS = 2_232_748              # MIND-large train release (behaviors.tsv)
EXPECTED_DEV_ROWS = 376_471                      # MIND-large dev release, used as test
TRAIN_FRACTION = 0.85

QC = []                                          # rows of data_quality_report.csv
def check(area, name, value, expected=None, status=None, split="all", detail=""):
    # record one check; status defaults to PASS/FAIL by comparing value with expected, INFO/WARN are set explicitly
    if status is None:
        status = "PASS" if (expected is None or value == expected) else "FAIL"
    QC.append({"check_id": f"{len(QC) + 1:03d}", "area": area, "split": split, "check": name,
               "value": value, "expected": "" if expected is None else expected, "status": status, "detail": detail})
    return value

def pct(a, b):
    return 100.0 * a / b if b else float("nan")

def fingerprint(path, head_bytes=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(head_bytes))
    return {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256_first_64MB": h.hexdigest()[:16]}

print("Setup ok ·", RUN_STAMP, "· polars", pl.__version__, "· pandas", pd.__version__)
""")

md(r"""
## 1. Inputs and the split procedure

`download_split_process.ipynb` reads the MIND-large **train** release (`behaviors.tsv`, 2,232,748 impressions, 9 to 14 Nov 2019),
parses the timestamps, sorts by time and cuts at row `int(0.85 × rows)`: the first 85 % become `train`, the rest `validation`.
The MIND-large **dev** release (15 Nov 2019, labelled) is kept whole as `test`. The news files for train and validation are the
rows of `news.tsv` referenced by that split's histories or slates; the test news file is the dev release's own `news.tsv`.

The checks below re-derive each of those statements from the saved parquet files.
""")

code(r"""
t0 = time.time()
BEHAVIOR_COLS = ["session_id", "user_id", "timestamp", "history", "impressions"]
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
B, N, inventory = {}, {}, []
for sp in SPLITS:
    bp, npath = ROOT / sp / f"behaviors_{sp}.parquet", ROOT / sp / f"news_{sp}.parquet"
    df = pl.read_parquet(bp)
    missing = [c for c in BEHAVIOR_COLS if c not in df.columns]
    check("schema", "behaviour columns present " + str(BEHAVIOR_COLS), missing, [], split=sp)
    if df["timestamp"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("timestamp").str.to_datetime(TIMESTAMP_FORMAT, strict=False))
    check("schema", "timestamp stored as datetime", str(df["timestamp"].dtype).startswith("Datetime"), True, split=sp)
    B[sp] = df.select(BEHAVIOR_COLS)
    nf = pl.read_parquet(npath)
    missing = [c for c in NEWS_COLS if c not in nf.columns]
    check("schema", "news columns present " + str(NEWS_COLS), missing, [], split=sp)
    N[sp] = nf.select(NEWS_COLS).with_columns(pl.all().cast(pl.Utf8))
    inventory += [{**fingerprint(bp), "rows": df.height}, {**fingerprint(npath), "rows": nf.height}]
inventory = pd.DataFrame(inventory)
display(inventory)

# ---- split arithmetic
n_tr, n_va, n_te = B["train"].height, B["validation"].height, B["test"].height
log_rows = n_tr + n_va
split_index = int(log_rows * TRAIN_FRACTION)
check("split", "train + validation rows = MIND-large train release", log_rows, EXPECTED_TRAIN_LOG_ROWS)
check("split", f"train rows = int({TRAIN_FRACTION} x train-release rows)", n_tr, split_index)
check("split", "validation rows = remaining rows", n_va, log_rows - split_index)
check("split", "test rows = MIND-large dev release", n_te, EXPECTED_DEV_ROWS)

# ---- time windows
win = pd.DataFrame([{"split": sp, "rows": B[sp].height, "first": B[sp]["timestamp"].min(), "last": B[sp]["timestamp"].max(),
                     "sorted_by_time_in_file": bool(B[sp]["timestamp"].is_sorted()),
                     "null_timestamps": int(B[sp]["timestamp"].null_count())} for sp in SPLITS])
display(win)
for r in win.itertuples():
    check("split", "time window", f"{r.first} -> {r.last}", status="INFO", split=r.split)
    check("split", "null timestamps", r.null_timestamps, 0, split=r.split)
check("split", "last train timestamp <= first validation timestamp", bool(win.loc[0, "last"] <= win.loc[1, "first"]), True)
check("split", "last validation timestamp < first test timestamp", bool(win.loc[1, "last"] < win.loc[2, "first"]), True)
check("split", "test impressions all on 15 Nov 2019",
      int((B["test"]["timestamp"].dt.date() != pd.Timestamp("2019-11-15").date()).sum()), 0)
check("split", "train+validation impressions all between 9 and 14 Nov 2019",
      int(pl.concat([B["train"]["timestamp"], B["validation"]["timestamp"]]).dt.date()
          .is_between(pd.Timestamp("2019-11-09").date(), pd.Timestamp("2019-11-14").date()).not_().sum()), 0)

# ---- boundary ties: the cut falls inside one second; same-second rows sit on both sides
boundary = B["train"]["timestamp"].max()
ties_tr = int((B["train"]["timestamp"] == boundary).sum()); ties_va = int((B["validation"]["timestamp"] == boundary).sum())
check("split", f"rows at the boundary second {boundary}: train / validation", f"{ties_tr} / {ties_va}", status="INFO",
      detail="the saved team split used the original timestamp sort; a stable rebuild in download_split_process.ipynb may assign boundary rows "
             "differently; the saved parquet files are the reference. No leakage: both sides share one timestamp.")
# a stable time sort of the saved files reproduces the saved assignment exactly
tv = pl.concat([B["train"].select("session_id", "timestamp").with_columns(split=pl.lit("train")),
                B["validation"].select("session_id", "timestamp").with_columns(split=pl.lit("validation"))])
resorted = tv.sort("timestamp", maintain_order=True)
check("split", "stable re-sort of the saved rows reproduces the 85/15 assignment (mismatching rows)",
      int((resorted.slice(0, split_index)["split"] != "train").sum()) + int((resorted.slice(split_index)["split"] != "validation").sum()), 0)

# ---- identifiers
for sp in SPLITS:
    check("keys", "session_id unique within the split", bool(B[sp]["session_id"].n_unique() == B[sp].height), True, split=sp)
check("keys", "session_id overlap between train and validation (count)",
      int(B["train"]["session_id"].is_in(B["validation"]["session_id"].implode()).sum()), 0)
overlap_te = int(B["test"]["session_id"].is_in(pl.concat([B["train"]["session_id"], B["validation"]["session_id"]]).implode()).sum())
check("keys", "session_id values of test that also occur in train/validation", overlap_te, status="WARN" if overlap_te else "PASS",
      detail="the dev release numbers impressions from 1 again; session_id alone is not a key across splits. "
             "Processed files add impression_uid = split_code * 10,000,000 + session_id.")
for sp in SPLITS:
    check("keys", "file sorted by timestamp", bool(win.set_index("split").loc[sp, "sorted_by_time_in_file"]),
          status="INFO", split=sp, detail="processed impressions are written sorted by (timestamp, session_id)")
print(f"Loaded in {time.time() - t0:.0f} s")
""")

md(r"""
## 2. Parsing the impression lists and histories

Each `impressions` string is a space-separated list of `N<id>-<label>` tokens (one per candidate shown; the order was shuffled
by Microsoft before release). Each `history` string is a space-separated list of `N<id>` tokens (articles the user clicked before
the log window), or null when the user has no recorded history. Both are parsed here and the derived counts are kept on the
impression table.
""")

code(r"""
t0 = time.time()
IMP = {}
for sp in SPLITS:
    df = B[sp]
    n = df.height
    for c in ["session_id", "user_id", "timestamp", "impressions"]:
        check("missing", f"null values in {c}", int(df[c].null_count()), 0, split=sp)
    check("missing", "blank impression strings", int((df["impressions"].str.strip_chars() == "").sum()), 0, split=sp)
    check("grammar", "impression strings not matching ^N<id>-<0|1>( N<id>-<0|1>)*$",
          int(df.select((pl.col("impressions").str.count_matches(r"^N\d+-[01]( N\d+-[01])*$") != 1).sum()).item()), 0, split=sp)
    check("grammar", "non-null history strings not matching ^N<id>( N<id>)*$",
          int(df.select((pl.col("history").is_not_null() & (pl.col("history").str.count_matches(r"^N\d+( N\d+)*$") != 1)).sum()).item()),
          0, split=sp)
    imp = (df.with_columns(
            split=pl.lit(sp), split_code=pl.lit(SPLIT_CODE[sp], dtype=pl.Int8),
            slate_len=(pl.col("impressions").str.count_matches(" ") + 1).cast(pl.Int16),
            n_clicks=(pl.col("impressions") + " ").str.count_matches("-1 ").cast(pl.Int16),
            history_len=pl.when(pl.col("history").is_null() | (pl.col("history") == "")).then(0)
                          .otherwise(pl.col("history").str.count_matches(" ") + 1).cast(pl.Int16),
            distinct_in_slate=pl.col("impressions").str.split(" ").list.eval(pl.element().str.head(-2)).list.n_unique().cast(pl.Int16),
            history_distinct=pl.when(pl.col("history").is_null()).then(0)
                               .otherwise(pl.col("history").str.split(" ").list.n_unique()).cast(pl.Int16))
          .with_columns(n_negatives=(pl.col("slate_len") - pl.col("n_clicks")).cast(pl.Int16),
                        impression_uid=(pl.col("split_code").cast(pl.Int64) * 10_000_000 + pl.col("session_id").cast(pl.Int64))))
    IMP[sp] = imp
    exposures = int(imp["slate_len"].cast(pl.Int64).sum()); clicks = int(imp["n_clicks"].cast(pl.Int64).sum())
    check("counts", "impressions", n, status="INFO", split=sp)
    check("counts", "exposures (candidate pairs = sum of slate lengths)", exposures, status="INFO", split=sp)
    check("counts", "clicks", clicks, status="INFO", split=sp)
    check("counts", "overall CTR (%)", round(pct(clicks, exposures), 4), status="INFO", split=sp)
    check("labels", "impressions with zero clicks", int((imp["n_clicks"] == 0).sum()), status="INFO", split=sp,
          detail="the release keeps only impressions with at least one click; CTR is click-conditioned (see impression_analysis)")
    check("labels", "slate length min / median / max",
          f"{int(imp['slate_len'].min())} / {int(imp['slate_len'].median())} / {int(imp['slate_len'].max())}", status="INFO", split=sp)
    check("duplicates", "impressions with the same article twice in one slate",
          int((imp["distinct_in_slate"] != imp["slate_len"]).sum()), 0, split=sp)
    n_null_hist = int(imp["history"].null_count())
    check("missing", "impressions with no recorded history (null)", f"{n_null_hist} ({pct(n_null_hist, n):.2f}%)", status="INFO", split=sp,
          detail="cold-start rows; kept, encoded as history_len = 0")
    check("missing", "blank (non-null, empty) history strings", int((imp["history"] == "").sum()), 0, split=sp)
    rep_entries = int((imp["history_len"] - imp["history_distinct"]).cast(pl.Int64).sum())
    users_rep = int(imp.filter(pl.col("history_len") != pl.col("history_distinct"))["user_id"].n_unique())
    check("duplicates", "repeated article ids inside a history (entries, counted per impression)",
          f"{rep_entries} entries in {users_rep} users", status="INFO", split=sp,
          detail="a user can click the same article twice before the window; kept as supplied")
    check("history", "history length median / p95 / max",
          f"{int(imp['history_len'].median())} / {int(imp['history_len'].quantile(0.95))} / {int(imp['history_len'].max())}",
          status="INFO", split=sp)
summary = pd.DataFrame([{"split": sp, "impressions": IMP[sp].height, "users": IMP[sp]["user_id"].n_unique(),
                         "exposures": int(IMP[sp]["slate_len"].cast(pl.Int64).sum()), "clicks": int(IMP[sp]["n_clicks"].cast(pl.Int64).sum()),
                         "ctr_pct": pct(int(IMP[sp]["n_clicks"].cast(pl.Int64).sum()), int(IMP[sp]["slate_len"].cast(pl.Int64).sum())),
                         "zero_click_impressions": int((IMP[sp]["n_clicks"] == 0).sum()),
                         "median_slate_len": float(IMP[sp]["slate_len"].median()), "max_slate_len": int(IMP[sp]["slate_len"].max()),
                         "null_history_pct": pct(int(IMP[sp]["history"].null_count()), IMP[sp].height),
                         "median_history_len": float(IMP[sp]["history_len"].median())} for sp in SPLITS])
display(summary.round(3))
print(f"Parsed in {time.time() - t0:.0f} s")
""")

md(r"""
## 3. Links between behaviour and news records

Every article id that appears in a history or a slate must exist in that split's news file, and (by construction of the split
notebook) every news row should be referenced by at least one history or slate.
""")

code(r"""
t0 = time.time()
def referenced_ids(df):
    hist = (df.lazy().select(pl.col("history").drop_nulls().str.split(" ")).explode("history")
              .filter(pl.col("history") != "").select(pl.col("history").unique()).collect()["history"])
    slate = (df.lazy().select(pl.col("impressions").str.split(" ")).explode("impressions")
               .select(pl.col("impressions").str.head(-2).unique().alias("n")).collect()["n"])
    return set(hist.to_list()), set(slate.to_list())

REF = {}
link_rows = []
for sp in SPLITS:
    H_ids, S_ids = referenced_ids(B[sp]); REF[sp] = (H_ids, S_ids)
    news_ids = set(N[sp]["news_id"].to_list())
    bad_fmt = int((N[sp]["news_id"].str.count_matches(r"^N\d+$") != 1).sum())
    check("links", "news_id values not matching ^N<digits>$", bad_fmt, 0, split=sp)
    check("links", "duplicate news_id rows in the news file", N[sp].height - len(news_ids), 0, split=sp)
    check("links", "slate article ids missing from the news file", len(S_ids - news_ids), 0, split=sp)
    check("links", "history article ids missing from the news file", len(H_ids - news_ids), 0, split=sp)
    check("links", "news rows never referenced by a history or a slate", len(news_ids - (H_ids | S_ids)), 0, split=sp,
          detail="news_<split> is exactly the set of referenced articles" if not (news_ids - (H_ids | S_ids)) else "")
    link_rows.append({"split": sp, "news_rows": N[sp].height, "articles_in_histories": len(H_ids), "articles_in_slates": len(S_ids),
                      "in_both": len(H_ids & S_ids), "history_only": len(H_ids - S_ids), "slate_only": len(S_ids - H_ids)})
links = pd.DataFrame(link_rows); display(links)
print(f"Checked in {time.time() - t0:.0f} s")
""")

md(r"""
## 4. News-field quality: text, entity annotations, entity embeddings

A null or blank string is *missing*. For the two entity columns a literal `[]` is a *valid empty list* (the article has no linked
entity) and is counted separately from missing or unparseable JSON. Entity embeddings (`entity_embedding.vec`, 100-d TransE vectors
keyed by Wikidata id) ship with the train and dev releases; the validation split is part of the train release and shares its file.
""")

code(r"""
t0 = time.time()
def is_blank_expr(c):
    return pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")

def parse_entities(value):
    if value is None or not str(value).strip():
        return None, "missing"
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None, "invalid"
    if not isinstance(parsed, list) or not all(isinstance(e, dict) for e in parsed):
        return None, "invalid"
    return parsed, ("empty" if not parsed else "present")

def load_entity_vectors(path, expected_dim=100):
    ids, bad = set(), 0
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != expected_dim + 1:
                bad += 1; continue
            ids.add(parts[0])
    return ids, bad

VEC = {"train": ROOT / "train" / "entity_embedding.vec", "validation": ROOT / "train" / "entity_embedding.vec",
       "test": ROOT / "test" / "entity_embedding.vec"}
vec_cache = {}
quality_rows, ENTITY_IDS = [], {}
for sp in SPLITS:
    nf = N[sp]; n = nf.height
    blanks = {c: int(nf.select(is_blank_expr(c).sum()).item()) for c in NEWS_COLS}
    for c in ["news_id", "category", "subcategory", "title", "url"]:
        check("news fields", f"blank {c}", blanks[c], 0, split=sp)
    check("news fields", "blank abstract", f"{blanks['abstract']} ({pct(blanks['abstract'], n):.2f}%)", status="INFO", split=sp,
          detail="kept as null; title is always present, so text models fall back to the title")
    dup_titles = int(nf["title"].is_duplicated().sum())
    check("news fields", "rows whose title also appears on another article (syndication / repeats)",
          f"{dup_titles} ({pct(dup_titles, n):.2f}%)", status="INFO", split=sp)
    check("news fields", "distinct categories / subcategories", f"{nf['category'].n_unique()} / {nf['subcategory'].n_unique()}", status="INFO", split=sp)
    ids_all = set()
    for col in ["title_entities", "abstract_entities"]:
        status = Counter(); counts = []
        for v in nf[col].to_list():
            parsed, st = parse_entities(v); status[st] += 1
            if parsed:
                counts.append(len(parsed)); ids_all.update(e.get("WikidataId") for e in parsed if e.get("WikidataId"))
        check("entities", f"{col}: unparseable JSON", status["invalid"], 0, split=sp)
        check("entities", f"{col}: missing field", status["missing"], 0, split=sp)
        check("entities", f"{col}: valid empty list []", f"{status['empty']} ({pct(status['empty'], n):.2f}%)", status="INFO", split=sp,
              detail="[] means no linked entity, it is not a missing value")
        quality_rows.append({"split": sp, "field": col, "present": status["present"], "empty_list": status["empty"],
                             "missing": status["missing"], "invalid": status["invalid"],
                             "coverage_pct": pct(status["present"], n), "mean_entities_when_present": float(np.mean(counts)) if counts else 0.0})
    ENTITY_IDS[sp] = ids_all
    if VEC[sp].is_file():
        if VEC[sp] not in vec_cache:
            vec_cache[VEC[sp]] = load_entity_vectors(VEC[sp])
        vec_ids, bad_rows = vec_cache[VEC[sp]]
        covered = len(ids_all & vec_ids)
        check("entities", f"entity_embedding.vec rows with a wrong dimension ({VEC[sp].relative_to(ROOT)})", bad_rows, 0, split=sp)
        check("entities", "annotated Wikidata ids with an embedding vector",
              f"{covered} of {len(ids_all)} ({pct(covered, len(ids_all)):.2f}%)", status="INFO", split=sp,
              detail="ids without a vector need an unknown-vector or mask in the representation notebook")
    else:
        check("entities", "entity_embedding.vec present", False, status="WARN", split=sp)
quality = pd.DataFrame(quality_rows); display(quality.round(2))
print(f"Audited in {time.time() - t0:.0f} s")
""")

md(r"""
## 5. Temporal integrity: can future information reach an earlier impression?

Four things have to hold for a time-based split to be honest.

* **Windows are ordered** (checked in section 1): every train impression precedes every validation impression, which precedes every test impression.
* **Histories are static snapshots.** The `history` string must not grow inside the window; if it did, a click made at 14:00 could
  appear in the history of the same user's 10:00 impression. We check that a user's history is identical across all of their
  impressions, within a split and across splits (including the dev-release test file).
* **Nothing in a split's news file depends on later behaviour.** Article metadata is static, and section 3 showed that each news file
  is exactly the set of articles referenced by its own split.
* **Overlap is known.** Users and articles shared between splits are not leakage, but they decide what "cold start" means for the
  model; the numbers are recorded here and analysed in `2_EDA/user_analysis.ipynb` and `article_temporal_analysis.ipynb`.
""")

code(r"""
t0 = time.time()
hist_all = pl.concat([IMP[sp].select("user_id", "history", "split") for sp in SPLITS])
per_user = hist_all.group_by("user_id").agg(n_hist=pl.col("history").n_unique(), n_split=pl.col("split").n_unique(), n_imp=pl.len())
n_users_all = per_user.height
check("temporal", "distinct users across the three splits", n_users_all, status="INFO")
check("temporal", "users whose history string differs between two of their impressions (any split)", int((per_user["n_hist"] > 1).sum()), 0,
      detail="history is a static pre-window snapshot: in-window clicks never enter it")
for sp in SPLITS:
    pu = IMP[sp].group_by("user_id").agg(n_hist=pl.col("history").n_unique(), n_imp=pl.len())
    check("temporal", "users with > 1 impression / of which history changed",
          f"{int((pu['n_imp'] > 1).sum())} / {int(((pu['n_imp'] > 1) & (pu['n_hist'] > 1)).sum())}", status="INFO", split=sp)
multi = per_user.filter(pl.col("n_split") > 1)
check("temporal", "users present in more than one split", multi.height, status="INFO")
tr_u, va_u, te_u = (set(IMP[sp]["user_id"].unique().to_list()) for sp in SPLITS)
check("temporal", "validation users not seen in train", f"{len(va_u - tr_u)} ({pct(len(va_u - tr_u), len(va_u)):.1f}%)", status="INFO", split="validation")
check("temporal", "test users not seen in train", f"{len(te_u - tr_u)} ({pct(len(te_u - tr_u), len(te_u)):.1f}%)", status="INFO", split="test")
check("temporal", "test users not seen in train or validation", f"{len(te_u - tr_u - va_u)} ({pct(len(te_u - tr_u - va_u), len(te_u)):.1f}%)", status="INFO", split="test")
check("temporal", "users in both train and test", len(tr_u & te_u), status="INFO")
# the dev release carries the same snapshot: a shared user's test history equals the train history
j = (IMP["train"].select("user_id", "history").unique("user_id").rename({"history": "h_train"})
     .join(IMP["test"].select("user_id", "history").unique("user_id").rename({"history": "h_test"}), on="user_id", how="inner"))
same = int((j["h_train"].fill_null("") == j["h_test"].fill_null("")).sum())
check("temporal", "shared train/test users whose test history differs from their train history", j.height - same, 0,
      detail="the user's clicks logged on 9-14 Nov are NOT in the test-day history; models may append them time-causally (see user_analysis)")
# in-window clicks that are also in the history of the same user (legitimate repeats, not leakage)
Htr, Str = REF["train"]; Hva, Sva = REF["validation"]; Hte, Ste = REF["test"]
check("temporal", "train slate articles that also appear in some train history", f"{len(Str & Htr)} of {len(Str)}", status="INFO", split="train",
      detail="a history article can be re-shown inside the window; the pair-level overlap (0.3% of candidates) is analysed in impression_analysis")
check("temporal", "validation slate articles also shown in train slates", f"{len(Sva & Str)} of {len(Sva)} ({pct(len(Sva & Str), len(Sva)):.1f}%)", status="INFO", split="validation")
check("temporal", "test slate articles also shown in train slates", f"{len(Ste & Str)} of {len(Ste)} ({pct(len(Ste & Str), len(Ste)):.1f}%)", status="INFO", split="test",
      detail="more than half of the test-day slate articles are new: popularity features must handle unseen articles")
check("temporal", "test slate articles present in the train news file in any role", f"{len(Ste & set(N['train']['news_id'].to_list()))} of {len(Ste)}", status="INFO", split="test")
check("temporal", "timezone of timestamps", "undocumented by Microsoft (log clock); hour-of-day features use the log clock", status="INFO")
overlap_tbl = pd.DataFrame([
    {"split": "validation", "users": len(va_u), "new_users_vs_train": len(va_u - tr_u), "new_users_pct": pct(len(va_u - tr_u), len(va_u)),
     "slate_articles": len(Sva), "slate_articles_seen_in_train_pct": pct(len(Sva & Str), len(Sva))},
    {"split": "test", "users": len(te_u), "new_users_vs_train": len(te_u - tr_u), "new_users_pct": pct(len(te_u - tr_u), len(te_u)),
     "slate_articles": len(Ste), "slate_articles_seen_in_train_pct": pct(len(Ste & Str), len(Ste))}])
display(overlap_tbl.round(1))
print(f"Checked in {time.time() - t0:.0f} s")
""")

md(r"""
## 6. Cleaning decisions and processed files

| # | Decision | Reason |
|---|---|---|
| 1 | **No impression is dropped.** All 1,897,835 / 334,913 / 376,471 impressions are kept. | Every row passed the grammar, label and duplicate checks; dropping rows would change the agreed split. |
| 2 | **Empty histories are kept** (`history_len = 0`). | They are the cold-start users the system must serve; 2 % of impressions. |
| 3 | **Candidates that are already in the user's history are kept.** | 0.3 % of candidates; removing them would alter slates and they are a legitimate repeat-exposure signal. |
| 4 | **Null abstracts are kept as null**, never imputed here. | 5.3 % of articles; the representation notebook decides the fallback (title only). |
| 5 | **`[]` entity lists are kept as valid empty lists.** | They mean "no linked entity", which is information, not missingness. |
| 6 | **Rows are sorted by (timestamp, session_id)** in the processed files. | The dev release is not time-ordered; sorted files make time-causal features cheap. |
| 7 | **A cross-split key `impression_uid`** = `split_code × 10,000,000 + session_id` is added. | `session_id` restarts at 1 in the dev release. |
| 8 | **`user_code` and `news_code`** integer keys are added (`news_code` = digits of `news_id`; `user_code` = rank of `user_id` over all three splits). | Compact joins on 97 million candidate rows. |
| 9 | **Impression strings are exploded** into one row per (impression, candidate) with the 1-based `position`. | `position` is the shuffled file order, kept only for the artefact analysis; it is not a model input. |
| 10 | **Article first/last exposure times and exposure counts per split** are stored on the catalogue. | Convenience for freshness and popularity work; they are retrospective and must be used only for events after `first_exposure_*`. |

Processed files (written below):

* `processed/users.parquet` — `user_id`, `user_code`.
* `processed/impressions_<split>.parquet` — one row per impression: `impression_uid, split, session_id, user_id, user_code, timestamp, history, history_len, slate_len, n_clicks, n_negatives`.
* `processed/candidates_<split>.parquet` — one row per candidate: `impression_uid, user_code, timestamp, position, news_code, clicked`.
* `processed/news_catalogue.parquet` — union of the three news files with split flags, text and entity counts, and per-split exposure facts.
""")

code(r"""
t0 = time.time()
users = (pl.concat([IMP[sp].select("user_id") for sp in SPLITS]).unique().sort("user_id")
         .with_row_index("user_code").with_columns(pl.col("user_code").cast(pl.Int32)).select("user_id", "user_code"))
users.write_parquet(PROCESSED / "users.parquet")
check("outputs", "users.parquet rows", users.height, n_users_all)

IMP_COLS = ["impression_uid", "split", "session_id", "user_id", "user_code", "timestamp", "history", "history_len", "slate_len", "n_clicks", "n_negatives"]
article_facts = []
written = []
for sp in SPLITS:
    imp = (IMP[sp].join(users, on="user_id", how="left").sort("timestamp", "session_id"))
    imp.select(IMP_COLS).write_parquet(PROCESSED / f"impressions_{sp}.parquet")
    cand = (imp.lazy().select("impression_uid", "user_code", "timestamp", pl.col("impressions").str.split(" ").alias("tok"))
            .with_columns(position=pl.int_ranges(1, pl.col("tok").list.len() + 1, dtype=pl.Int16))
            .explode("tok", "position")
            .select("impression_uid", "user_code", "timestamp", "position",
                    news_code=pl.col("tok").str.head(-2).str.strip_prefix("N").cast(pl.Int32),
                    clicked=pl.col("tok").str.tail(1).cast(pl.Int8))
            .collect(engine="streaming"))
    exp_rows = int(imp["slate_len"].cast(pl.Int64).sum()); exp_clicks = int(imp["n_clicks"].cast(pl.Int64).sum())
    check("outputs", "candidate rows = sum of slate lengths", cand.height, exp_rows, split=sp)
    check("outputs", "candidate clicks = sum of impression clicks", int(cand["clicked"].cast(pl.Int64).sum()), exp_clicks, split=sp)
    check("outputs", "candidate labels outside {0,1}", int((~cand["clicked"].is_in([0, 1])).sum()), 0, split=sp)
    cand.write_parquet(PROCESSED / f"candidates_{sp}.parquet")
    article_facts.append(
        cand.group_by("news_code").agg(pl.col("timestamp").min().alias(f"first_exposure_{sp}"), pl.col("timestamp").max().alias(f"last_exposure_{sp}"),
                                       pl.len().cast(pl.Int64).alias(f"exposures_{sp}"), pl.col("clicked").cast(pl.Int64).sum().alias(f"clicks_{sp}")))
    written.append({"file": f"candidates_{sp}.parquet", "rows": cand.height}); written.append({"file": f"impressions_{sp}.parquet", "rows": imp.height})
    del cand, imp

# ---- catalogue: union of the three news files
cat = None
for sp in SPLITS:
    flag = N[sp].select("news_id").with_columns(pl.lit(True).alias(f"in_news_{sp}"))
    cat = flag if cat is None else cat.join(flag, on="news_id", how="full", coalesce=True)
cat = cat.with_columns([pl.col(f"in_news_{sp}").fill_null(False) for sp in SPLITS])
meta = pl.concat([N[sp] for sp in SPLITS]).unique(subset=["news_id"], keep="first")
cat = (cat.join(meta, on="news_id", how="left")
       .with_columns(news_code=pl.col("news_id").str.strip_prefix("N").cast(pl.Int32),
                     title_len_words=pl.col("title").fill_null("").str.count_matches(r"\S+").cast(pl.Int16),
                     abstract_len_words=pl.col("abstract").fill_null("").str.count_matches(r"\S+").cast(pl.Int16),
                     has_abstract=~is_blank_expr("abstract"),
                     n_title_entities=pl.col("title_entities").fill_null("").str.count_matches('"WikidataId"').cast(pl.Int16),
                     n_abstract_entities=pl.col("abstract_entities").fill_null("").str.count_matches('"WikidataId"').cast(pl.Int16)))
for af in article_facts:
    cat = cat.join(af, on="news_code", how="left")
cat = cat.with_columns([pl.col(f"exposures_{sp}").fill_null(0) for sp in SPLITS] + [pl.col(f"clicks_{sp}").fill_null(0) for sp in SPLITS])
cat = cat.select(["news_id", "news_code", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
                 + [f"in_news_{sp}" for sp in SPLITS]
                 + ["title_len_words", "abstract_len_words", "has_abstract", "n_title_entities", "n_abstract_entities"]
                 + [c for sp in SPLITS for c in (f"first_exposure_{sp}", f"last_exposure_{sp}", f"exposures_{sp}", f"clicks_{sp}")]).sort("news_code")
cat.write_parquet(PROCESSED / "news_catalogue.parquet")
check("outputs", "news_catalogue rows = distinct articles across the three news files",
      cat.height, len(set().union(*[set(N[sp]['news_id'].to_list()) for sp in SPLITS])))
check("outputs", "news_code unique", bool(cat["news_code"].n_unique() == cat.height), True)
written.append({"file": "news_catalogue.parquet", "rows": cat.height}); written.append({"file": "users.parquet", "rows": users.height})
sizes = pd.DataFrame(written); sizes["MB"] = [round((PROCESSED / f).stat().st_size / 1e6, 1) for f in sizes.file]
display(sizes)
display(cat.select("news_id", "news_code", "category", "subcategory", "in_news_train", "in_news_test", "title_len_words", "abstract_len_words",
                   "n_title_entities", "first_exposure_train", "exposures_train", "clicks_train", "exposures_test").head(5).to_pandas())
print(f"Processed files written in {time.time() - t0:.0f} s")
""")

md(r"""
## 7. Data quality report

`data_quality_report.csv` lists every check above. `PASS` / `FAIL` rows compare a value with an expectation; `INFO` rows record a
fact that later notebooks rely on; `WARN` rows are caveats that need handling downstream (none of them required dropping data).
""")

code(r"""
report = pd.DataFrame(QC)
report.insert(0, "run_stamp", RUN_STAMP)
report.to_csv(REPORT_PATH, index=False)
counts = report["status"].value_counts().reindex(["PASS", "FAIL", "WARN", "INFO"]).fillna(0).astype(int)
print("Checks:", dict(counts), "->", REPORT_PATH)
if counts["FAIL"]:
    display(report[report.status == "FAIL"])
    raise ValueError(f"{counts['FAIL']} checks failed; see data_quality_report.csv")
display(report[report.status.isin(["WARN"])][["check_id", "area", "split", "check", "value", "detail"]])
display(report[report.area.isin(["split", "temporal"])][["check_id", "split", "check", "value", "expected", "status"]])

summary_md = f'''
### Summary for the report

* **Split verified.** Train = first {TRAIN_FRACTION:.0%} of the time-sorted MIND-large train release ({n_tr:,} impressions, 9 Nov 00:00 to 14 Nov 07:30),
  validation = the rest of 14 Nov ({n_va:,}), test = the dev release ({n_te:,}, 15 Nov). The windows do not overlap; the only subtlety is
  {ties_tr + ties_va} same-second rows at the cut, and a stable re-sort reproduces the saved assignment exactly.
* **Grammar and labels are clean.** Every impression string parses as `N<id>-<0|1>`, labels are binary, no slate repeats an article,
  no impression has zero clicks (the release is click-conditioned), and no article id is missing from its news file.
* **Missingness is limited to abstracts (5.3 %) and histories (2 % of impressions).** Entity columns have no invalid JSON; about a quarter of
  articles carry a valid empty entity list. Entity embeddings cover about 97 % of annotated Wikidata ids.
* **No future information enters earlier impressions.** Histories are frozen snapshots (0 users with a changing history, test histories
  identical to train histories), and every news file depends only on its own split.
* **Overlap facts for modelling.** {pct(len(va_u - tr_u), len(va_u)):.1f} % of validation users and {pct(len(te_u - tr_u), len(te_u)):.1f} % of test users are new;
  {pct(len(Ste & Str), len(Ste)):.1f} % of test-day slate articles were already shown in train, so cold-start articles dominate the test day.
'''
display(Markdown(summary_md))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
