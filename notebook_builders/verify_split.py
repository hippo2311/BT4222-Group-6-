import time

import polars as pl

t0 = time.time()
B = {
    s: pl.read_parquet(f"{s}/behaviors_{s}.parquet")
    for s in ["train", "validation", "test"]
}
N = {
    s: pl.read_parquet(f"{s}/news_{s}.parquet") for s in ["train", "validation", "test"]
}
print("loaded", round(time.time() - t0, 1), "s")
for s, df in B.items():
    print(
        f"\n== {s}: rows={df.height:,} cols={df.columns} dtypes={[str(d) for d in df.dtypes]}"
    )
    print("  time", df["timestamp"].min(), "->", df["timestamp"].max())
    print(
        "  session_id unique:",
        df["session_id"].n_unique() == df.height,
        " nulls:",
        df.null_count().to_dicts()[0],
    )
    print(
        "  history null:",
        df["history"].is_null().sum(),
        " history empty-string:",
        (df["history"] == "").sum(),
    )
    print(
        "  impressions null/empty:",
        df["impressions"].is_null().sum(),
        (df["impressions"] == "").sum(),
    )
    # sorted?
    print("  sorted by time:", df["timestamp"].is_sorted())
# 1. Reconstruct split from concatenated train+val
tv = pl.concat(
    [
        B["train"].with_columns(pl.lit("train").alias("split")),
        B["validation"].with_columns(pl.lit("validation").alias("split")),
    ]
)
print(
    "\n== train+val rows",
    tv.height,
    " expected 2,232,748; 85% index =",
    int(tv.height * 0.85),
)
print(
    "  session_id unique across train+val:",
    tv["session_id"].n_unique() == tv.height,
    " min/max id",
    tv["session_id"].min(),
    tv["session_id"].max(),
)
# any val row earlier than any train row?
print(
    "  max train ts <= min val ts:",
    B["train"]["timestamp"].max() <= B["validation"]["timestamp"].min(),
)
bts = B["train"]["timestamp"].max()
nb_tr = (B["train"]["timestamp"] == bts).sum()
nb_va = (B["validation"]["timestamp"] == bts).sum()
print(
    f"  boundary timestamp {bts}: {nb_tr} rows in train, {nb_va} in validation (tie split)"
)
# 2. Original order check: does train+val sorted by timestamp reproduce exactly the 85% cut? (ties make the exact assignment order-dependent)
srt = tv.sort("timestamp", maintain_order=True)
cut = int(srt.height * 0.85)
mism = (srt.slice(0, cut)["split"] != "train").sum()
print("  rows that would differ under a stable re-sort of the parquet order:", mism)
# 3. User overlap
U = {s: set(B[s]["user_id"].unique().to_list()) for s in B}
print(
    "\n== users: train",
    len(U["train"]),
    "val",
    len(U["validation"]),
    "test",
    len(U["test"]),
)
print(
    "  val users unseen in train:",
    len(U["validation"] - U["train"]),
    f"({len(U['validation'] - U['train']) / len(U['validation']):.1%})",
)
print(
    "  test users unseen in train:",
    len(U["test"] - U["train"]),
    f"({len(U['test'] - U['train']) / len(U['test']):.1%})",
)
print(
    "  test users unseen in train+val:", len(U["test"] - U["train"] - U["validation"])
)


# 4. news files: referenced ids per split
def ref_ids(df):
    h = df.select(pl.col("history").fill_null("").str.split(" ").explode()).filter(
        pl.col("history") != ""
    )["history"]
    i = df.select(
        pl.col("impressions")
        .str.split(" ")
        .explode()
        .str.extract(r"^([^-]+)", 1)
        .alias("x")
    )["x"]
    return set(h.unique().to_list()), set(i.unique().to_list())


REF = {}
for s, behaviors in B.items():
    H, I = ref_ids(behaviors)
    ref = H | I
    nid = set(N[s]["news_id"].to_list())
    print(
        f"\n== {s} news: file rows {N[s].height:,}, unique ids {len(nid):,}, referenced ids {len(ref):,} (history {len(H):,}, impressions {len(I):,}, both {len(H & I):,})"
    )
    print(
        "   referenced but missing from news file:",
        len(ref - nid),
        "  in news file but never referenced:",
        len(nid - ref),
    )
    print(
        "   news_id duplicates:",
        N[s].height - len(nid),
        " null title:",
        N[s]["title"].is_null().sum(),
        " null abstract:",
        N[s]["abstract"].is_null().sum(),
    )
    REF[s] = (H, I)
# 5. leakage checks: test/val slates vs train catalogue
Htr, Itr = REF["train"]
Hva, Iva = REF["validation"]
Hte, Ite = REF["test"]
print("\n== catalogue overlap")
print(
    "  val impression articles also shown in train:",
    len(Iva & Itr),
    "of",
    len(Iva),
    f"({len(Iva & Itr) / len(Iva):.1%})",
)
print(
    "  test impression articles also shown in train:",
    len(Ite & Itr),
    "of",
    len(Ite),
    f"({len(Ite & Itr) / len(Ite):.1%})",
)
print(
    "  test impression articles in train news file (any role):",
    len(Ite & set(N["train"]["news_id"])),
    f"({len(Ite & set(N['train']['news_id'])) / len(Ite):.1%})",
)
print("  test history articles shown in train slates:", len(Hte & Itr), "of", len(Hte))
# do test histories contain articles first seen in val/train window (i.e. histories were refreshed for the dev set)?
print(
    "  test history articles that were impression candidates in train:",
    len(Hte & Itr),
    " in val:",
    len(Hte & Iva),
)
print(
    "  train history articles that are impression candidates in train:",
    len(Htr & Itr),
    " (history = pre-window snapshot if small)",
)
# 6. per-user history static across splits?
hist = pl.concat(
    [
        B[s].select("user_id", "history").with_columns(pl.lit(s).alias("split"))
        for s in B
    ]
)
per_user = hist.group_by("user_id").agg(
    pl.col("history").n_unique().alias("nh"), pl.col("split").n_unique().alias("ns")
)
print(
    "\n== users with >1 distinct history string overall:",
    (per_user["nh"] > 1).sum(),
    " of",
    per_user.height,
)
multi = per_user.filter(pl.col("ns") > 1)
print(
    "  users in >1 split:",
    multi.height,
    " of which history changed:",
    (multi["nh"] > 1).sum(),
)
# within train+val only
pu2 = (
    hist.filter(pl.col("split") != "test")
    .group_by("user_id")
    .agg(pl.col("history").n_unique().alias("nh"))
)
print("  within train+val, users with changed history:", (pu2["nh"] > 1).sum())
# test vs train history for shared users
tr_h = (
    B["train"]
    .select("user_id", "history")
    .unique("user_id")
    .rename({"history": "h_train"})
)
te_h = (
    B["test"]
    .select("user_id", "history")
    .unique("user_id")
    .rename({"history": "h_test"})
)
j = tr_h.join(te_h, on="user_id", how="inner")
print(
    "  shared train/test users:",
    j.height,
    " same history string:",
    (j["h_train"] == j["h_test"]).sum(),
    " test history longer:",
    (
        j["h_test"].fill_null("").str.len_chars()
        > j["h_train"].fill_null("").str.len_chars()
    ).sum(),
)
# 7. duplicates within slate / label validity
for s, behaviors in B.items():
    ex = behaviors.select(
        pl.col("session_id"), pl.col("impressions").str.split(" ").alias("t")
    ).explode("t")
    bad = ex.filter(~pl.col("t").str.contains(r"^N\d+-[01]$")).height
    dup = (
        ex.with_columns(pl.col("t").str.extract(r"^([^-]+)", 1).alias("n"))
        .group_by("session_id", "n")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    nclick = behaviors.select(pl.col("impressions").str.count_matches("-1").alias("c"))[
        "c"
    ]
    print(
        f"\n== {s}: malformed tokens {bad}, duplicate article within slate {dup}, impressions with 0 clicks {(nclick == 0).sum()}, exposures {ex.height:,}, clicks {nclick.sum():,}"
    )
print("\ndone", round(time.time() - t0, 1), "s")
