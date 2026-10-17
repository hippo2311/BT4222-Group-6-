from __future__ import annotations

import sqlite3
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DEFAULT_SETTINGS, normalised_mean

HOUR = 3600000000000


class PairStore:
    """Disk-backed user/article history; avoids a 97M-row Python dictionary.

    A fresh replay must use a fresh database. The persistent database is a scratch
    backend, not a complete replay checkpoint (profile/article state is separate).
    """

    def __init__(self, path=":memory:"):
        if str(path) != ":memory:":
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(
                    f"Pair state already exists: {path}. Use a fresh replay path."
                )
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA cache_size=-32768")
        self.db.execute(
            "CREATE TABLE pairs(u INTEGER,a INTEGER,e INTEGER,c INTEGER,last INTEGER,PRIMARY KEY(u,a)) WITHOUT ROWID"
        )

    def lookup(self, users, articles):
        result = []
        pairs = list(zip(map(int, users), map(int, articles)))
        for start in range(0, len(pairs), 300):
            chunk = pairs[start : start + 300]
            sql = (
                "WITH wanted(i,u,a) AS (VALUES "
                + ",".join(["(?,?,?)"] * len(chunk))
                + ") SELECT COALESCE(p.e,0),COALESCE(p.c,0),p.last FROM wanted w LEFT JOIN pairs p ON p.u=w.u AND p.a=w.a ORDER BY w.i"
            )
            result.extend(
                self.db.execute(
                    sql, [v for i, pair in enumerate(chunk) for v in (i, *pair)]
                ).fetchall()
            )
        return result

    def update(self, users, articles, labels, t):
        grouped = Counter()
        for u, a, y in zip(users, articles, labels):
            grouped[int(u), int(a), "e"] += 1
            grouped[int(u), int(a), "c"] += int(y)
        keys = {(u, a) for u, a, _ in grouped}
        rows = [
            (u, a, grouped[u, a, "e"], grouped[u, a, "c"], int(t.value))
            for u, a in sorted(keys)
        ]
        self.db.executemany(
            "INSERT INTO pairs VALUES(?,?,?,?,?) ON CONFLICT(u,a) DO UPDATE SET e=e+excluded.e,c=c+excluded.c,last=excluded.last",
            rows,
        )
        self.db.commit()

    def close(self):
        self.db.close()


@dataclass
class User:
    snapshot: np.ndarray
    snapshot_key: tuple
    observed: deque
    categories: Counter = field(default_factory=Counter)
    subcategories: Counter = field(default_factory=Counter)
    n_observed: int = 0
    last_impression: pd.Timestamp | None = None
    version: int = 0


class TemporalState:
    def __init__(self, bundle, start, end, cfg=DEFAULT_SETTINGS, pair_path=":memory:"):
        self.bundle = bundle
        self.cfg = cfg
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.start_hour = self.start.value // HOUR
        hours = int(self.end.value // HOUR - self.start_hour) + 1
        if hours < 1:
            raise ValueError("End before start")
        self.exposures = np.zeros(bundle.n, np.int64)
        self.clicks = np.zeros(bundle.n, np.int64)
        self.first = np.full(bundle.n, -1, np.int64)
        self.hour_exposures = np.zeros((bundle.n, hours), np.int32)
        self.hour_clicks = np.zeros((bundle.n, hours), np.int32)
        self.cat = np.concatenate(
            [
                ["<PAD>"],
                bundle.index.category.fillna("<MISSING>").astype(str).to_numpy(),
            ]
        )
        self.subcat = np.concatenate(
            [
                ["<PAD>"],
                bundle.index.subcategory.fillna("<MISSING>").astype(str).to_numpy(),
            ]
        )
        self.users = {}
        self.profiles = {}
        self.pairs = PairStore(pair_path)
        self.total_exposures = 0
        self.total_clicks = 0
        self.last_update = None
        self.active_time = None

    def begin(self, requests, t):
        t = pd.Timestamp(t)
        if self.active_time is not None:
            raise ValueError("Previous timestamp has not been finished")
        if self.last_update is not None and t <= self.last_update:
            raise ValueError(
                "Replay timestamps must increase strictly; merge same-time requests"
            )
        if t < self.start or t > self.end:
            raise ValueError("Request outside state horizon")
        if (
            requests.impression_uid.duplicated().any()
            or not requests.timestamp.eq(t).all()
        ):
            raise ValueError("begin requires unique requests from this timestamp only")
        self.active_time = t
        for row in requests.itertuples(index=False):
            text = "" if pd.isna(row.history) else str(row.history)
            codes = tuple(int(token.removeprefix("N")) for token in text.split())
            key = int(row.user_code)
            if key in self.users:
                if self.users[key].snapshot_key != codes:
                    raise ValueError(f"Supplied history changed for user {key}")
                continue
            snapshot = self.bundle.rows(codes)
            user = User(snapshot, codes, deque(maxlen=self.cfg.history_cap))
            user.categories.update(self.cat[snapshot])
            user.subcategories.update(self.subcat[snapshot])
            self.users[key] = user

    def profile(self, user_code, t):
        t = pd.Timestamp(t)
        if self.active_time != t:
            raise ValueError(
                "Call begin() for this timestamp before requesting profiles"
            )
        key = int(user_code)
        user = self.users[key]
        cache_key = (key, user.version, t.value)
        if cache_key in self.profiles:
            return self.profiles[cache_key]
        observed = list(user.observed)
        seed = (
            user.snapshot
            if self.cfg.snapshot_order_verified
            else np.sort(user.snapshot)
        )
        history = np.r_[seed, [a for _, a in observed]].astype(np.int32)[
            -self.cfg.history_cap :
        ]
        recent = [
            a for time, a in observed if t.value - time <= self.cfg.recent_hours * HOUR
        ][-self.cfg.recent_count :]
        text, valid_text = normalised_mean(self.bundle.text, history)
        entity, valid_entity = normalised_mean(self.bundle.entity, history)
        long, _ = normalised_mean(self.bundle.text, user.snapshot)
        recent_vector, valid_recent = normalised_mean(self.bundle.text, recent)
        interests = []
        interest_categories = []
        for cat, _ in sorted(user.categories.items(), key=lambda x: (-x[1], x[0]))[
            : self.cfg.interest_count
        ]:
            rows = history[self.cat[history] == cat]
            vector, valid = normalised_mean(self.bundle.title, rows)
            if valid and np.linalg.norm(vector) > 0:
                interests.append(vector)
                interest_categories.append(cat)
        padded = np.zeros(self.cfg.history_cap, np.int32)
        padded[: len(history)] = history
        result = {
            "text": text,
            "entity": entity,
            "long_text": long,
            "recent_text": recent_vector,
            "interests": np.asarray(interests, np.float32).reshape(
                -1, self.cfg.text_dim
            ),
            "history_rows": padded,
            "history_mask": padded != 0,
            "selected_interest_categories": interest_categories,
            "recent_rows": list(map(int, recent)),
            "history_length": len(user.snapshot) + user.n_observed,
            "retained_length": len(history),
            "valid_text_fraction": valid_text / max(1, len(history)),
            "valid_entity_fraction": valid_entity / max(1, len(history)),
            "has_text": bool(np.linalg.norm(text) > 0),
            "has_entity": bool(np.linalg.norm(entity) > 0),
            "has_recent": bool(valid_recent and np.linalg.norm(recent_vector) > 0),
            "no_supplied_history": len(user.snapshot) == 0,
            "group_count": len(interests),
            "selected_group_count": min(self.cfg.interest_count, len(user.categories)),
            "state_version": user.version,
        }
        self.profiles[cache_key] = result
        return result

    def half_life(self, rows, t):
        t = pd.Timestamp(t)
        recent_end = int(t.value // HOUR - self.start_hour)
        recent_start = recent_end - 6
        result = []
        for row in rows:
            first = self.first[row]
            if first < 0:
                result.append((0.0, False, "no_prior_exposure", 0, 0))
                continue
            early_start = int((first + HOUR - 1) // HOUR - self.start_hour)
            early_end = early_start + 6
            if recent_start < early_end:
                result.append((0.0, False, "young_or_overlapping", 0, 0))
                continue
            early_e = int(self.hour_exposures[row, early_start:early_end].sum())
            recent_e = int(self.hour_exposures[row, recent_start:recent_end].sum())
            if min(early_e, recent_e) < self.cfg.half_life_min_exposures:
                result.append((0.0, False, "low_support", early_e, recent_e))
                continue
            early_c = int(self.hour_clicks[row, early_start:early_end].sum())
            recent_c = int(self.hour_clicks[row, recent_start:recent_end].sum())
            early = (early_c + 1) / (early_e + 2)
            recent = (recent_c + 1) / (recent_e + 2)
            if recent >= early:
                result.append((0.0, False, "no_decline", early_e, recent_e))
                continue
            estimate = np.log(2) * (recent_start - early_start) / np.log(early / recent)
            ok = bool(np.isfinite(estimate) and estimate > 0)
            result.append(
                (
                    float(estimate) if ok else 0.0,
                    ok,
                    "available" if ok else "invalid",
                    early_e,
                    recent_e,
                )
            )
        return result

    def finish(self, requests, events, t, feedback_splits):
        t = pd.Timestamp(t)
        if self.active_time != t:
            raise ValueError("Timestamp is not active")
        if not events.timestamp.eq(t).all() or not set(events.impression_uid) <= set(
            requests.impression_uid
        ):
            raise ValueError("Feedback must belong to this timestamp's requests")
        owners = requests.set_index("impression_uid").user_code
        if not np.array_equal(
            events.user_code.to_numpy(), events.impression_uid.map(owners).to_numpy()
        ):
            raise ValueError("Feedback/request user mismatch")
        if events.duplicated(["impression_uid", "news_code"]).any():
            raise ValueError("Duplicate feedback event")
        for u in requests.user_code.unique():
            self.users[int(u)].last_impression = t
        selected = events[events.split.isin(feedback_splits)]
        if len(selected):
            if not selected.clicked.isin([0, 1]).all():
                raise ValueError("Feedback labels must be binary")
            rows = self.bundle.rows(selected.news_code)
            labels = selected.clicked.to_numpy(np.int64)
            self.pairs.update(selected.user_code, rows, labels, t)
            np.add.at(self.exposures, rows, 1)
            np.add.at(self.clicks, rows, labels)
            unseen = self.first[rows] < 0
            self.first[rows[unseen]] = t.value
            hour = int(t.value // HOUR - self.start_hour)
            np.add.at(self.hour_exposures[:, hour], rows, 1)
            np.add.at(self.hour_clicks[:, hour], rows, labels)
            self.total_exposures += len(rows)
            self.total_clicks += int(labels.sum())
            clicked = selected.loc[selected.clicked.eq(1)].sort_values(
                ["user_code", "news_code", "impression_uid"]
            )
            for event in clicked.itertuples(index=False):
                user = self.users[int(event.user_code)]
                a = self.bundle.lookup[int(event.news_code)]
                user.observed.append((t.value, a))
                user.n_observed += 1
                user.categories[self.cat[a]] += 1
                user.subcategories[self.subcat[a]] += 1
                user.version += 1
        self.profiles.clear()
        self.last_update = t
        self.active_time = None

    def close(self):
        self.pairs.close()
