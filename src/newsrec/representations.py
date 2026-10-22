from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import (
    DEFAULT_SETTINGS,
    SPLITS,
    manifest,
    measure,
    normalise,
    require_inputs,
    sha256,
    upsert_report,
    write_json,
)

STATIC_COLUMNS = [
    "news_id",
    "news_code",
    "category",
    "subcategory",
    "title",
    "abstract",
    "title_entities",
    "abstract_entities",
]


def artifact_dir(root):
    path = Path(root) / "3_Representations" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(value):
    import unicodedata

    return " ".join(
        unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value)).split()
    )


def read_catalogue(root):
    path = Path(root) / "1_Data_Preparation" / "processed" / "news_catalogue.parquet"
    df = pd.read_parquet(path)
    required = STATIC_COLUMNS + ["in_news_train"]
    if not set(required) <= set(df):
        raise ValueError(f"Catalogue lacks {set(required) - set(df)}")
    df = df.sort_values("news_code").reset_index(drop=True)
    if df.news_code.duplicated().any() or df.news_id.duplicated().any():
        raise ValueError("Article IDs must be unique")
    return df


def audit_metadata_conflicts(root):
    """Compare raw split metadata when available; preparation keeps first occurrence."""
    pieces = []
    for split in SPLITS:
        p = Path(root) / split / f"news_{split}.parquet"
        if p.exists():
            frame = pd.read_parquet(p, columns=STATIC_COLUMNS[:1] + STATIC_COLUMNS[2:])
            frame["source"] = split
            pieces.append(frame)
    if len(pieces) < 2:
        return {"status": "raw metadata not available for cross-release comparison"}
    joined = pd.concat(pieces, ignore_index=True).fillna("")
    conflicts = {
        c: int((joined.groupby("news_id")[c].nunique() > 1).sum())
        for c in STATIC_COLUMNS[2:]
    }
    return {
        "status": "checked; prepared catalogue retains deterministic source precedence",
        "conflicting_article_counts": conflicts,
    }


def fit_text_space(news, cfg=DEFAULT_SETTINGS):
    """Fit vocabulary/IDF and SVD on training titles; freeze for every transform."""
    titles = news.title.map(clean_text)
    abstracts = news.abstract.map(clean_text)
    train = news.in_news_train.fillna(False).to_numpy(dtype=bool)
    if train.sum() < 2:
        raise ValueError("At least two training articles are required")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        dtype=np.float32,
    )
    train_tfidf = tfidf.fit_transform(titles[train])
    components = min(cfg.text_dim, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1)
    if components < 1:
        raise ValueError(
            "Insufficient training vocabulary for SVD; inspect min_df/text"
        )
    svd = TruncatedSVD(n_components=components, random_state=cfg.seed)
    svd.fit(train_tfidf)
    matrices = []
    for texts in (titles, abstracts):
        reduced = normalise(svd.transform(tfidf.transform(texts)))
        dense = np.zeros((len(titles) + 1, cfg.text_dim), np.float32)
        dense[1:, :components] = reduced
        matrices.append(dense)
    title, abstract = matrices
    tv = np.linalg.norm(title, axis=1) > 0
    av = np.linalg.norm(abstract, axis=1) > 0
    text = normalise(cfg.title_weight * title + (1 - cfg.title_weight) * abstract)
    text[tv & ~av] = title[tv & ~av]
    text[av & ~tv] = abstract[av & ~tv]
    return tfidf, svd, components, title, abstract, text, tv, av


def encode_title_tokens(titles, train, cfg=DEFAULT_SETTINGS):
    tokenise = TfidfVectorizer().build_analyzer()
    frequency = Counter(token for text_ in titles[train] for token in tokenise(text_))
    token_vocab = {
        w: i + 2
        for i, (w, _) in enumerate(
            sorted(frequency.items(), key=lambda x: (-x[1], x[0]))[: cfg.max_features]
        )
    }
    token_ids = np.zeros((len(titles) + 1, cfg.title_tokens), np.int32)
    token_lengths, unknown, total = ([], 0, 0)
    for i, text_ in enumerate(titles, 1):
        tokens = tokenise(text_)
        token_lengths.append(len(tokens))
        ids = [token_vocab.get(w, 1) for w in tokens]
        unknown += sum(x == 1 for x in ids)
        total += len(ids)
        token_ids[i, : min(len(ids), cfg.title_tokens)] = ids[: cfg.title_tokens]
    return token_ids, token_vocab, token_lengths, unknown, total


def fit_category_encodings(news, train):
    encodings = {}
    for col in ("category", "subcategory"):
        vocabulary = sorted(set(news.loc[train, col].dropna().astype(str)))
        lookup = {v: i + 1 for i, v in enumerate(vocabulary)}
        codes = news[col].map(lookup).fillna(0).astype(np.int32)
        encodings[col] = (codes, lookup)
    return encodings


def build_news(
    root,
    cfg=DEFAULT_SETTINGS,
    *,
    text_encoder=fit_text_space,
    token_encoder=encode_title_tokens,
    category_encoder=fit_category_encodings,
):
    processed = require_inputs(root)
    out = artifact_dir(root)
    with measure(root, "text"):
        news = read_catalogue(root)
        titles = news.title.map(clean_text)
        abstracts = news.abstract.map(clean_text)
        train = news.in_news_train.fillna(False).to_numpy(dtype=bool)
        tfidf, svd, components, title, abstract, text, tv, av = text_encoder(news, cfg)
        index = news[STATIC_COLUMNS].copy()
        index.insert(0, "article_row", np.arange(1, len(news) + 1, dtype=np.int32))
        for col, (codes, lookup) in category_encoder(news, train).items():
            index[f"{col}_code"] = codes
            write_json(out / f"{col}_vocabulary.json", lookup)
        index["has_abstract"] = abstracts.ne("")
        index["title_word_count"] = titles.str.split().str.len().astype(np.int32)
        index["abstract_word_count"] = abstracts.str.split().str.len().astype(np.int32)
        index["has_title_vector"] = tv[1:]
        index["has_abstract_vector"] = av[1:]
        index["has_text_vector"] = np.linalg.norm(text[1:], axis=1) > 0
        index.to_parquet(out / "article_index.parquet", index=False)
        for name, matrix in (
            ("title64", title),
            ("abstract64", abstract),
            ("text64", text),
        ):
            np.save(out / f"{name}.npy", matrix)
        with (out / "text_transform.pkl").open("wb") as fh:
            pickle.dump({"tfidf": tfidf, "svd": svd}, fh)
        token_ids, token_vocab, token_lengths, unknown, total = token_encoder(
            titles, train, cfg
        )
        np.save(out / "title_token_ids.npy", token_ids)
        np.save(out / "title_token_mask.npy", token_ids != 0)
        write_json(
            out / "title_vocabulary.json",
            {"padding_id": 0, "unknown_id": 1, "tokens": token_vocab},
        )
        coverage = []
        for split in SPLITS:
            mask = news[f"in_news_{split}"].to_numpy(bool)
            for rep, valid in (
                ("title", tv[1:]),
                ("abstract", av[1:]),
                ("text", index.has_text_vector.to_numpy()),
            ):
                n = int(mask.sum())
                k = int(valid[mask].sum())
                coverage.append(
                    {
                        "split": split,
                        "representation": rep,
                        "cohort": "split_catalogue",
                        "denominator": n,
                        "valid": k,
                        "fallback": n - k,
                        "valid_fraction": k / n if n else 0,
                    }
                )
        upsert_report(
            root, "representation_coverage.csv", pd.DataFrame(coverage), "text"
        )
        outputs = [
            out / f"{name}.npy"
            for name in (
                "title64",
                "abstract64",
                "text64",
                "title_token_ids",
                "title_token_mask",
            )
        ] + [out / "article_index.parquet", out / "text_transform.pkl"]
        outputs += [
            out / f"{c}_vocabulary.json" for c in ("title", "category", "subcategory")
        ]
        write_json(
            out / "news_manifest.json",
            manifest(
                root,
                cfg,
                [processed / "news_catalogue.parquet"],
                outputs,
                padded_row=0,
                effective_svd_dim=components,
                stored_text_dim=cfg.text_dim,
                article_index_hash=sha256(out / "article_index.parquet"),
                fit_scope="unique training-release article titles only",
                token_oov_fraction=unknown / max(1, total),
                title_truncation_fraction=float(
                    np.mean(np.array(token_lengths) > cfg.title_tokens)
                ),
                metadata_audit=audit_metadata_conflicts(root),
            ),
        )
    return pd.DataFrame(coverage)


def read_entity_lookup(path, dim):
    ids, vectors = ([], [])
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != dim + 1:
                raise ValueError(f"{path}:{line_no}: expected ID + {dim} coordinates")
            if parts[0] in seen:
                raise ValueError(f"Duplicate entity {parts[0]}")
            vector = np.asarray(parts[1:], np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Nonfinite entity {parts[0]}")
            ids.append(parts[0])
            vectors.append(vector)
            seen.add(parts[0])
    if not ids:
        raise ValueError("Entity file is empty")
    return (ids, np.asarray(vectors, np.float32))


def parse_entities(value):
    if (
        value is None
        or (isinstance(value, float) and np.isnan(value))
        or (not str(value).strip())
    ):
        return ([], "missing")
    try:
        data = json.loads(value)
    except (ValueError, TypeError):
        return ([], "invalid")
    if not isinstance(data, list) or any(not isinstance(e, dict) for e in data):
        return ([], "invalid")
    ids = sorted({str(e["WikidataId"]) for e in data if e.get("WikidataId")})
    return (ids, "present" if data else "empty")


def pool_article_entities(index, lookup, vectors, cfg=DEFAULT_SETTINGS):
    article = np.zeros((len(index) + 1, cfg.entity_dim), np.float32)
    links, quality, article_sets = ([], [], [])
    for row in index.itertuples(index=False):
        title_ids, ts = parse_entities(row.title_entities)
        abstract_ids, aps = parse_entities(row.abstract_entities)
        all_ids = sorted(set(title_ids) | set(abstract_ids))
        article_sets.append(set(all_ids))
        matched = [
            lookup[x]
            for x in all_ids
            if x in lookup and np.linalg.norm(vectors[lookup[x]]) > 0
        ]
        if matched:
            article[row.article_row] = normalise(vectors[matched].mean(axis=0))
        quality.append(
            {
                "news_code": row.news_code,
                "title_entity_status": ts,
                "abstract_entity_status": aps,
                "n_title_entities": len(title_ids),
                "n_abstract_entities": len(abstract_ids),
                "n_annotated_entities": len(all_ids),
                "n_valid_entities": len(matched),
                "entity_coverage": len(matched) / max(1, len(all_ids)),
                "has_entity_vector": bool(np.linalg.norm(article[row.article_row]) > 0),
            }
        )
        for key in all_ids:
            links.append(
                {
                    "news_code": row.news_code,
                    "entity_id": key,
                    "entity_row": lookup.get(key, 0),
                    "in_title": key in title_ids,
                    "in_abstract": key in abstract_ids,
                }
            )
    return article, links, quality, article_sets


def build_entities(root, cfg=DEFAULT_SETTINGS, *, article_pooler=pool_article_entities):
    require_inputs(root)
    out = artifact_dir(root)
    news_manifest = json.loads((out / "news_manifest.json").read_text())
    if news_manifest["representation_config_hash"] != cfg.representation_fingerprint():
        raise ValueError("Rebuild news representations with this configuration first")
    with measure(root, "entity"):
        index = pd.read_parquet(out / "article_index.parquet")
        if sha256(out / "article_index.parquet") != news_manifest["article_index_hash"]:
            raise ValueError("Article index changed; rebuild text representations")
        source = Path(root) / "train" / "entity_embedding.vec"
        ids, raw = read_entity_lookup(source, cfg.entity_dim)
        lookup = {key: i + 1 for i, key in enumerate(ids)}
        vectors = np.vstack([np.zeros((1, cfg.entity_dim), np.float32), normalise(raw)])
        np.save(out / "entity_vectors.npy", vectors)
        pd.DataFrame(
            {"entity_id": ids, "entity_row": np.arange(1, len(ids) + 1)}
        ).to_parquet(out / "entity_index.parquet", index=False)
        article, links, quality, article_sets = article_pooler(
            index, lookup, vectors, cfg
        )
        np.save(out / "article_entity100.npy", article)
        pd.DataFrame(quality).to_parquet(out / "entity_quality.parquet", index=False)
        pd.DataFrame(
            links,
            columns=["news_code", "entity_id", "entity_row", "in_title", "in_abstract"],
        ).to_parquet(out / "article_entity_links.parquet", index=False)
        catalogue = read_catalogue(root)
        coverage = []
        for split in SPLITS:
            mask = catalogue[f"in_news_{split}"].to_numpy(bool)
            observed = set().union(*(article_sets[i] for i in np.flatnonzero(mask)))
            valid_ids = {
                x
                for x in observed
                if x in lookup and np.linalg.norm(vectors[lookup[x]]) > 0
            }
            for rep, n, k in (
                ("entity_ids", len(observed), len(valid_ids)),
                (
                    "article_entity",
                    int(mask.sum()),
                    int((np.linalg.norm(article[1:][mask], axis=1) > 0).sum()),
                ),
            ):
                coverage.append(
                    {
                        "split": split,
                        "representation": rep,
                        "cohort": "split_catalogue",
                        "denominator": n,
                        "valid": k,
                        "fallback": n - k,
                        "valid_fraction": k / max(1, n),
                    }
                )
        upsert_report(
            root, "representation_coverage.csv", pd.DataFrame(coverage), "entity"
        )
        compatibility = {
            "status": "dev lookup absent; canonical train space used throughout"
        }
        dev = Path(root) / "test" / "entity_embedding.vec"
        sources = [source, out / "article_index.parquet"]
        if dev.exists():
            dev_ids, dev_raw = read_entity_lookup(dev, cfg.entity_dim)
            dev_map = {k: i for i, k in enumerate(dev_ids)}
            shared = sorted(set(ids) & set(dev_ids))
            max_diff = max(
                (
                    float(np.max(np.abs(raw[lookup[k] - 1] - dev_raw[dev_map[k]])))
                    for k in shared
                ),
                default=None,
            )
            compatibility = {
                "shared_ids": len(shared),
                "max_coordinate_difference": max_diff,
                "dev_only_ids": len(set(dev_ids) - set(ids)),
                "policy": "audit only; no dev-only vectors merged",
            }
            sources.append(dev)
        outputs = [
            out / name
            for name in (
                "entity_vectors.npy",
                "entity_index.parquet",
                "article_entity100.npy",
                "entity_quality.parquet",
                "article_entity_links.parquet",
            )
        ]
        write_json(
            out / "entity_manifest.json",
            manifest(
                root,
                cfg,
                sources,
                outputs,
                article_index_hash=sha256(out / "article_index.parquet"),
                canonical_space="train",
                zero_lookup_vectors=int((np.linalg.norm(raw, axis=1) == 0).sum()),
                dev_compatibility=compatibility,
            ),
        )
    return pd.DataFrame(coverage)


def compute_availability(processed):
    histories = []
    exposures = []
    for split in SPLITS:
        histories.append(
            pl.scan_parquet(processed / f"impressions_{split}.parquet").select(
                "user_code", "timestamp", "history"
            )
        )
        exposures.append(
            pl.scan_parquet(processed / f"candidates_{split}.parquet").select(
                "news_code", "timestamp"
            )
        )
    first_hist = (
        pl.concat(histories)
        .group_by("user_code", "history")
        .agg(pl.col("timestamp").min())
        .filter(pl.col("history").is_not_null() & (pl.col("history") != ""))
        .with_columns(pl.col("history").str.split(" ").alias("news_id"))
        # Empty histories were filtered above. Do not pass the removed
        # ``empty_as_null`` keyword: recent Polars releases reject it on
        # LazyFrame.explode(), whereas the default explosion is sufficient here.
        .explode("news_id")
        .with_columns(
            pl.col("news_id").str.strip_prefix("N").cast(pl.Int32).alias("news_code")
        )
        .group_by("news_code")
        .agg(pl.col("timestamp").min().alias("history_discovered_at"))
    )
    first_exp = (
        pl.concat(exposures)
        .group_by("news_code")
        .agg(pl.col("timestamp").min().alias("first_observed_exposure"))
    )
    frame = (
        first_hist.join(first_exp, on="news_code", how="full", coalesce=True)
        .with_columns(
            pl.min_horizontal("history_discovered_at", "first_observed_exposure").alias(
                "available_from"
            )
        )
        .sort("news_code")
        .collect(engine="streaming")
        .to_pandas()
    )
    frame["discovery_source"] = np.where(
        frame.history_discovered_at.isna(),
        "slate",
        np.where(
            frame.first_observed_exposure.isna()
            | (frame.history_discovered_at < frame.first_observed_exposure),
            "history",
            "slate_or_tied",
        ),
    )
    start = min(
        pl.scan_parquet(processed / f"impressions_{s}.parquet")
        .select(pl.col("timestamp").min())
        .collect()
        .item()
        for s in SPLITS
    )
    frame["left_censored"] = frame.first_observed_exposure.eq(start)
    return frame, start


def build_availability(
    root, cfg=DEFAULT_SETTINGS, *, availability_builder=compute_availability
):
    """Discovery from earliest observable history/slate, never an invented publication date."""
    processed = require_inputs(root)
    out = artifact_dir(root)
    with measure(root, "availability"):
        frame, start = availability_builder(processed)
        frame.to_parquet(out / "article_availability.parquet", index=False)
        if (out / "article_index.parquet").exists() and (
            out / "article_entity100.npy"
        ).exists():
            index = pd.read_parquet(out / "article_index.parquet").set_index(
                "news_code"
            )
            audit = frame.sort_values("available_from").copy()
            ordered_rows = index.loc[audit.news_code, "article_row"].to_numpy(np.int32)
            text_valid = (
                np.linalg.norm(
                    np.load(out / "text64.npy", mmap_mode="r")[ordered_rows], axis=1
                )
                > 0
            )
            entity_valid = (
                np.linalg.norm(
                    np.load(out / "article_entity100.npy", mmap_mode="r")[ordered_rows],
                    axis=1,
                )
                > 0
            )
            text_cum = np.r_[0, np.cumsum(text_valid)]
            entity_cum = np.r_[0, np.cumsum(entity_valid)]
            times = pd.to_datetime(audit.available_from).to_numpy(
                dtype="datetime64[ns]"
            )
            checkpoints = pd.date_range(
                pd.Timestamp(start).floor("h"),
                pd.Timestamp(frame.available_from.max()).ceil("h"),
                freq="h",
            )
            counts = np.searchsorted(
                times, checkpoints.to_numpy(dtype="datetime64[ns]"), side="right"
            )
            report = pd.DataFrame(
                {
                    "timestamp": checkpoints,
                    "eligible_articles": counts,
                    "eligible_text_vectors": text_cum[counts],
                    "eligible_entity_vectors": entity_cum[counts],
                }
            )
            report_path = Path(root) / "results" / "catalogue_coverage_over_time.csv"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(report_path, index=False)
        sources = [
            processed / f"{kind}_{s}.parquet"
            for kind in ("impressions", "candidates")
            for s in SPLITS
        ]
        write_json(
            out / "availability_manifest.json",
            manifest(
                root,
                cfg,
                sources,
                [out / "article_availability.parquet"],
                policy="minimum observable history/slate timestamp <= request; not publication time",
            ),
        )
    return frame


class Bundle:
    """Validated article-aligned arrays. Row 0 is padding/unknown, never a real article."""

    def __init__(self, root, cfg=DEFAULT_SETTINGS, verify_hashes=True):
        self.root = Path(root)
        self.cfg = cfg
        out = artifact_dir(root)
        index_hash = sha256(out / "article_index.parquet")
        for name in ("news", "entity", "availability"):
            meta = json.loads((out / f"{name}_manifest.json").read_text())
            if meta["representation_config_hash"] != cfg.representation_fingerprint():
                raise ValueError(
                    f"{name}: incompatible configuration; rebuild artifacts"
                )
            if meta.get("article_index_hash", index_hash) != index_hash:
                raise ValueError(
                    f"{name}: article map provenance mismatch; rebuild dependents"
                )
            if verify_hashes:
                for rel, expected in meta["outputs"].items():
                    if sha256(self.root / rel) != expected:
                        raise ValueError(f"Artifact hash mismatch: {rel}")
        self.index = pd.read_parquet(out / "article_index.parquet")
        self.lookup = dict(zip(self.index.news_code, self.index.article_row))
        self.n = len(self.index) + 1
        self.title = np.load(out / "title64.npy", mmap_mode="r")
        self.text = np.load(out / "text64.npy", mmap_mode="r")
        self.entity = np.load(out / "article_entity100.npy", mmap_mode="r")
        for array, dim in (
            (self.title, cfg.text_dim),
            (self.text, cfg.text_dim),
            (self.entity, cfg.entity_dim),
        ):
            norms = np.linalg.norm(array, axis=1)
            if (
                array.shape != (self.n, dim)
                or not np.isfinite(array).all()
                or np.any((norms > 1e-07) & (np.abs(norms - 1) > 0.0001))
                or (norms[0] != 0)
            ):
                raise ValueError(
                    "Invalid representation shape, coordinates, norm or padding"
                )
        self.meta = self.index.set_index("article_row").reindex(range(self.n))
        self.quality = pd.read_parquet(out / "entity_quality.parquet").set_index(
            "news_code"
        )
        self.availability = pd.read_parquet(
            out / "article_availability.parquet"
        ).set_index("news_code")
        if set(self.lookup) != set(self.availability.index):
            raise ValueError("Availability must cover exactly the prepared catalogue")
        self.category = np.r_[0, self.index.category_code.to_numpy(np.int32)]
        self.subcategory = np.r_[0, self.index.subcategory_code.to_numpy(np.int32)]

    def rows(self, codes):
        try:
            return np.asarray([self.lookup[int(c)] for c in codes], np.int32)
        except KeyError as exc:
            raise ValueError(f"Article absent from catalogue: {exc}") from exc

    def eligible(self, codes, t):
        return (
            self.availability.loc[list(codes), "available_from"] <= pd.Timestamp(t)
        ).to_numpy(bool)


def representation_summary(root, cfg=DEFAULT_SETTINGS):
    """Validate stored text arrays and return the compact notebook audit table."""
    out = artifact_dir(root)
    index = pd.read_parquet(out / "article_index.parquet")
    rows = []
    for name in ("title64", "abstract64", "text64"):
        array = np.load(out / f"{name}.npy", mmap_mode="r")
        norms = np.linalg.norm(array, axis=1)
        if (
            array.shape != (len(index) + 1, cfg.text_dim)
            or not np.isfinite(array).all()
            or np.any(array[0] != 0)
            or not np.allclose(norms[norms > 1e-7], 1, atol=1e-4)
        ):
            raise ValueError(f"Invalid stored text representation: {name}")
        rows.append(
            {
                "representation": name,
                "shape": str(array.shape),
                "MB": array.nbytes / 1e6,
                "valid_articles": int((norms[1:] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def text_neighbors(root, count=5):
    """One query article and its nearest valid text neighbours; no click labels."""
    if count < 1:
        raise ValueError("count must be positive")
    out = artifact_dir(root)
    index = pd.read_parquet(out / "article_index.parquet").set_index("article_row")
    vectors = np.load(out / "text64.npy", mmap_mode="r")
    valid = np.flatnonzero(np.linalg.norm(vectors, axis=1) > 0)
    if not len(valid):
        return pd.DataFrame(columns=["news_code", "title", "cosine", "is_query"])
    query = int(valid[0])
    scores = vectors @ vectors[query]
    candidates = valid[valid != query]
    neighbours = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
    selected = np.r_[query, neighbours]
    result = index.loc[selected, ["news_code", "title"]].copy()
    result["cosine"] = scores[selected]
    result["is_query"] = selected == query
    return result.reset_index()
