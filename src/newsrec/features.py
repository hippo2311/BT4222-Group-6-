from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .common import DEFAULT_SETTINGS
from .temporal import HOUR

GROUPS = {
    "content": [
        "category_code",
        "subcategory_code",
        "title_word_count",
        "abstract_word_count",
        "has_abstract",
        "n_annotated_entities",
        "entity_coverage",
        "article_has_text",
        "article_has_entity",
    ],
    "history": [
        "log_history_length",
        "retained_history_length",
        "valid_text_history_fraction",
        "valid_entity_history_fraction",
        "no_supplied_history",
        "no_causal_history",
        "no_text_profile",
        "no_entity_profile",
        "no_recent_profile",
    ],
    "match": [
        "category_affinity",
        "subcategory_affinity",
        "seen_category",
        "text_cosine",
        "entity_cosine",
        "long_term_cosine",
        "recent_cosine",
        "text_match_valid",
        "entity_match_valid",
        "long_term_match_valid",
        "recent_match_valid",
    ],
    "repeat": [
        "log_prior_exposure_count",
        "log_prior_nonclick_count",
        "log_prior_pair_click_count",
        "hours_since_last_exposure",
        "no_prior_pair_exposure",
        "previously_clicked",
    ],
    "popularity": [
        "log_prior_article_exposures",
        "log_prior_article_clicks",
        "smoothed_prior_article_ctr",
        "article_low_support",
        "population_startup",
    ],
    "freshness": [
        "article_age_hours",
        "no_prior_article_exposure",
        "age_left_censored",
    ],
    "multi_interest": [
        "multi_interest_match_score",
        "interest_group_count",
        "selected_interest_group_count",
        "multi_interest_valid",
    ],
    "half_life": [
        "estimated_article_half_life_hours",
        "has_half_life",
        "stale_after_grace",
    ],
    "request_context": [
        "day_part",
        "weekday",
        "hours_since_previous_impression",
        "first_visit",
    ],
    "slate_context": ["personalized_relative_advantage", "slate_context_valid"],
}


FORMULAS = {
    "category_affinity": "full prior clicked category count / full history count; zero if empty",
    "subcategory_affinity": "full prior clicked subcategory count / full history count; zero if empty",
    "seen_category": "prior candidate-category count > 0",
    "smoothed_prior_article_ctr": "(prior article clicks + strength * expanding Beta(1,1) global CTR) / (prior article exposures + strength)",
    "article_age_hours": "hours since first prior observed exposure; zero with flag if none",
    "estimated_article_half_life_hours": "ln(2) * completed 6h window-centre gap / ln(early CTR / recent CTR); zero plus mask when unsupported",
    "multi_interest_match_score": "max title64 cosine over up to three valid category interest vectors",
    "personalized_relative_advantage": "candidate multi-interest match minus actual candidate-set median; zero if context invalid",
}


def feature_schema(cfg=DEFAULT_SETTINGS):
    features = []
    for group, names in GROUPS.items():
        if group == "slate_context" and (not cfg.context_features):
            continue
        for name in names:
            features.append(
                {
                    "name": name,
                    "dtype": "float32",
                    "group": group,
                    "missing_value": 0,
                    "definition": FORMULAS.get(name, name.replace("_", " ")),
                    "cutoff": "static train-fitted content or events strictly before request timestamp",
                    "preprocessing": "log1p already applied"
                    if name.startswith("log_")
                    else "none; fit any scaler on training rows only",
                    "consumer": "ranker; slate context requires complete actual candidate set"
                    if group == "slate_context"
                    else "ranker",
                }
            )
    return {
        "version": "ranking-features-v1",
        "features": features,
        "model_columns": [f["name"] for f in features],
        "keys": ["impression_uid", "news_code"],
        "diagnostics": [
            "user_code",
            "article_row",
            "timestamp",
            "state_version",
            "candidate_source",
            "eligible",
            "history_length",
            "prior_article_exposures",
            "prior_article_clicks",
            "prior_exposure_count",
            "half_life_reason",
            "half_life_early_support",
            "half_life_recent_support",
        ],
        "excluded": [
            "clicked",
            "position",
            "slate_len",
            "n_clicks",
            "n_negatives",
            "split",
            "user_code",
            "news_code",
            "impression_uid",
        ],
        "update_mode": cfg.update_mode,
        "config_hash": cfg.fingerprint(),
    }


def build_features(
    requests,
    candidate_pairs,
    state_before_t,
    representation_bundle,
    feature_settings=DEFAULT_SETTINGS,
    candidate_source="retrieved",
):
    """Same input contract for logged or unlabelled retrieved candidates.

    Caller begins one complete timestamp, builds all feature requests, then calls
    finish with logged feedback. This function neither reads labels nor mutates state.
    """
    state = state_before_t
    bundle = representation_bundle
    cfg = feature_settings
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
    prior_pairs = state.pairs.lookup(pairs.user_code, rows)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    unique_half = state.half_life(unique_rows, t)
    half = [unique_half[i] for i in inverse]
    global_prior = (state.total_clicks + 1) / (state.total_exposures + 2)
    n = len(pairs)
    metadata = bundle.meta.loc[rows].reset_index(drop=True)
    quality = bundle.quality.loc[pairs.news_code.to_numpy()].reset_index(drop=True)
    article_e = state.exposures[rows]
    article_c = state.clicks[rows]
    first = state.first[rows]
    seen = first >= 0
    pair_e = np.asarray([x[0] for x in prior_pairs], np.int64)
    pair_c = np.asarray([x[1] for x in prior_pairs], np.int64)
    pair_last = np.asarray(
        [-1 if x[2] is None else x[2] for x in prior_pairs], np.int64
    )
    h = np.asarray([x[0] for x in half], np.float64)
    has_h = np.asarray([x[1] for x in half], bool)
    text_valid = np.linalg.norm(bundle.text[rows], axis=1) > 0
    entity_valid = np.linalg.norm(bundle.entity[rows], axis=1) > 0
    title_valid = np.linalg.norm(bundle.title[rows], axis=1) > 0
    ages = np.where(seen, (t.value - first) / HOUR, 0)
    values = {
        "impression_uid": pairs.impression_uid.to_numpy(np.int64),
        "news_code": pairs.news_code.to_numpy(np.int64),
        "user_code": pairs.user_code.to_numpy(np.int64),
        "article_row": rows,
        "timestamp": t,
        "candidate_source": candidate_source,
        "eligible": True,
        "category_code": metadata.category_code.to_numpy(),
        "subcategory_code": metadata.subcategory_code.to_numpy(),
        "title_word_count": metadata.title_word_count.to_numpy(),
        "abstract_word_count": metadata.abstract_word_count.to_numpy(),
        "has_abstract": metadata.has_abstract.to_numpy(),
        "n_annotated_entities": quality.n_annotated_entities.to_numpy(),
        "entity_coverage": quality.entity_coverage.to_numpy(),
        "article_has_text": text_valid,
        "article_has_entity": entity_valid,
        "prior_exposure_count": pair_e,
        "log_prior_exposure_count": np.log1p(pair_e),
        "log_prior_nonclick_count": np.log1p(pair_e - pair_c),
        "log_prior_pair_click_count": np.log1p(pair_c),
        "hours_since_last_exposure": np.where(
            pair_last >= 0, (t.value - pair_last) / HOUR, 0
        ),
        "no_prior_pair_exposure": pair_last < 0,
        "prior_article_exposures": article_e,
        "prior_article_clicks": article_c,
        "log_prior_article_exposures": np.log1p(article_e),
        "log_prior_article_clicks": np.log1p(article_c),
        "smoothed_prior_article_ctr": (article_c + cfg.ctr_strength * global_prior)
        / (article_e + cfg.ctr_strength),
        "article_low_support": article_e < 100,
        "population_startup": state.total_exposures == 0,
        "article_age_hours": ages,
        "no_prior_article_exposure": ~seen,
        "age_left_censored": first == state.start.value,
        "estimated_article_half_life_hours": h,
        "has_half_life": has_h,
        "stale_after_grace": has_h & (ages > h + 6),
        "half_life_reason": [x[2] for x in half],
        "half_life_early_support": [x[3] for x in half],
        "half_life_recent_support": [x[4] for x in half],
        "day_part": t.hour // 6,
        "weekday": t.weekday(),
    }
    user_columns = [
        "state_version",
        "history_length",
        "log_history_length",
        "retained_history_length",
        "valid_text_history_fraction",
        "valid_entity_history_fraction",
        "no_supplied_history",
        "no_causal_history",
        "no_text_profile",
        "no_entity_profile",
        "no_recent_profile",
        "category_affinity",
        "subcategory_affinity",
        "seen_category",
        "text_cosine",
        "entity_cosine",
        "long_term_cosine",
        "recent_cosine",
        "previously_clicked",
        "multi_interest_match_score",
        "interest_group_count",
        "selected_interest_group_count",
        "multi_interest_valid",
        "hours_since_previous_impression",
        "first_visit",
        "text_match_valid",
        "entity_match_valid",
        "long_term_match_valid",
        "recent_match_valid",
    ]
    values.update({name: np.zeros(n, np.float64) for name in user_columns})
    for user_code, positions in pairs.groupby("user_code", sort=False).indices.items():
        idx = np.asarray(positions)
        article_rows = rows[idx]
        user = state.users[int(user_code)]
        p = state.profile(user_code, t)
        total = p["history_length"]
        category_counts = np.asarray(
            [user.categories[c] for c in state.cat[article_rows]]
        )
        subcategory_counts = np.asarray(
            [user.subcategories[c] for c in state.subcat[article_rows]]
        )
        dynamic = {
            "state_version": p["state_version"],
            "history_length": total,
            "log_history_length": np.log1p(total),
            "retained_history_length": p["retained_length"],
            "valid_text_history_fraction": p["valid_text_fraction"],
            "valid_entity_history_fraction": p["valid_entity_fraction"],
            "no_supplied_history": p["no_supplied_history"],
            "no_causal_history": total == 0,
            "no_text_profile": not p["has_text"],
            "no_entity_profile": not p["has_entity"],
            "no_recent_profile": not p["has_recent"],
            "category_affinity": category_counts / max(1, total),
            "subcategory_affinity": subcategory_counts / max(1, total),
            "seen_category": category_counts > 0,
            "text_cosine": bundle.text[article_rows] @ p["text"],
            "entity_cosine": bundle.entity[article_rows] @ p["entity"],
            "long_term_cosine": bundle.text[article_rows] @ p["long_text"],
            "recent_cosine": bundle.text[article_rows] @ p["recent_text"],
            "text_match_valid": p["has_text"] & text_valid[idx],
            "entity_match_valid": p["has_entity"] & entity_valid[idx],
            "long_term_match_valid": bool(np.linalg.norm(p["long_text"]) > 0)
            & text_valid[idx],
            "recent_match_valid": p["has_recent"] & text_valid[idx],
            "previously_clicked": (pair_c[idx] > 0)
            | np.isin(article_rows, user.snapshot),
            "multi_interest_match_score": np.max(
                bundle.title[article_rows] @ p["interests"].T, axis=1
            )
            if p["group_count"]
            else np.zeros(len(idx)),
            "interest_group_count": p["group_count"],
            "selected_interest_group_count": p["selected_group_count"],
            "multi_interest_valid": bool(p["group_count"]) & title_valid[idx],
            "hours_since_previous_impression": (
                t - user.last_impression
            ).total_seconds()
            / 3600
            if user.last_impression is not None
            else 0,
            "first_visit": user.last_impression is None,
        }
        for name, value in dynamic.items():
            values[name][idx] = value
    result = pd.DataFrame(values)
    result["state_version"] = result.state_version.astype(np.int64)
    result["history_length"] = result.history_length.astype(np.int64)
    if cfg.context_features:
        grouped = result.groupby("impression_uid")
        valid = grouped.multi_interest_valid.transform("all")
        result["slate_context_valid"] = valid
        result["personalized_relative_advantage"] = (
            result.multi_interest_match_score
            - grouped.multi_interest_match_score.transform("median")
        ).where(valid, 0)
    columns = feature_schema(cfg)["model_columns"]
    result[columns] = result[columns].astype(np.float32)
    if not np.isfinite(result[columns].to_numpy()).all():
        raise ValueError("Nonfinite model features")
    return result


def sample_training_groups(candidates, cfg=DEFAULT_SETTINGS):
    """Reproducible per-positive exposed negatives, independent of input row order."""
    groups = []
    selected = set()
    for uid, slate in candidates.groupby("impression_uid", sort=True):
        if slate.news_code.duplicated().any() or not slate.clicked.isin([0, 1]).all():
            raise ValueError("Sampling requires unique binary-labelled logged pairs")
        positives = sorted(map(int, slate.loc[slate.clicked.eq(1), "news_code"]))
        negatives = np.sort(
            slate.loc[slate.clicked.eq(0), "news_code"].to_numpy(np.int64)
        )
        if not positives:
            raise ValueError("Logged training impression has no positives")
        for positive in positives:
            seed = int.from_bytes(
                hashlib.sha256(f"{cfg.seed}:{int(uid)}:{positive}".encode()).digest()[
                    :8
                ],
                "little",
            )
            rng = np.random.default_rng(seed)
            sampled = sorted(
                map(
                    int,
                    rng.choice(
                        negatives,
                        min(cfg.negative_count, len(negatives)),
                        replace=False,
                    ),
                )
            )
            groups.append(
                {
                    "impression_uid": int(uid),
                    "positive_news_code": positive,
                    "negative_news_codes": sampled,
                }
            )
            selected.add((int(uid), positive))
            selected.update((int(uid), n) for n in sampled)
    return (
        pd.DataFrame(
            groups,
            columns=["impression_uid", "positive_news_code", "negative_news_codes"],
        ),
        selected,
    )
