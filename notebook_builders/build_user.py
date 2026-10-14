import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from eda_common import Cells, write_notebook, CLOSING

c = Cells()
c.md(r"""
# 2. EDA — User analysis: activity, histories, cold start and history-based signals

**BT4222 Group 6 · final project · task 3 (exploratory data analysis), notebook 2 of 3.** Owner: Member 3 (Anh Minh Hoang).
Sections 5 to 7 follow Lee Yong Rui's interim notebooks (history affinity, novelty, short- versus long-term interest).

Questions and the decisions they feed:

| # | Question | Decision it feeds |
|---|---|---|
| 1 | How active are users inside the window, and how long are their histories? | sequence-length caps for user encoders, how much in-window behaviour exists per user |
| 2 | Who are the cold-start users, and how do they behave? | fallback path for users without history; how many validation / test users are new |
| 3 | What exactly is the supplied history, and can it be extended with in-window clicks? | user-profile construction for validation and test, leakage rules |
| 4 | What do histories contain (categories, overlap with the window catalogue)? | whether history-based content features can be built from the news file |
| 5 | Does history-category affinity predict clicks? | `category_affinity` feature |
| 6 | Does the category have to be in the history at all (novelty)? | `seen_category` flag, treatment of cold users |
| 7 | Do users return to stable preferences after long gaps? | short-term versus long-term user representations |

Inputs: `1_Data_Preparation/processed/` (training split; validation and test impressions are read only for the cold-start counts in §2 and §3).
Outputs: `2_EDA/figures/usr_*.png`, `2_EDA/tables/usr_*.csv`, `2_EDA/manifest_user_analysis.json`.
""")
c.setup("user_analysis")

c.md(r"""
## 0. Load
""")
c.code(r"""
t0 = time.time()
IMP = {sp: pl.read_parquet(PROCESSED / f"impressions_{sp}.parquet", columns=["impression_uid", "user_code", "timestamp", "history_len", "slate_len", "n_clicks"])
       for sp in ["train", "validation", "test"]}
imp = IMP["train"]
cand = pl.read_parquet(PROCESSED / f"candidates_{SPLIT}.parquet", columns=["impression_uid", "user_code", "timestamp", "news_code", "clicked"])
news = pl.read_parquet(PROCESSED / "news_catalogue.parquet", columns=["news_code", "category", "subcategory", "in_news_train", "exposures_train"])
hist = pl.read_parquet(PROCESSED / f"impressions_{SPLIT}.parquet", columns=["impression_uid", "user_code", "history"])
H["n_impressions"], H["n_users"], H["n_exposures"], H["n_clicks"] = imp.height, int(imp["user_code"].n_unique()), cand.height, int(cand["clicked"].cast(pl.Int64).sum())
print(f"train: {H['n_impressions']:,} impressions, {H['n_users']:,} users, {H['n_exposures']:,} exposures · loaded in {time.time() - t0:.0f} s")
""")

c.md(r"""
## 1. User activity and history length

One row per user: impressions, clicks, active days and the length of the supplied history (identical across the user's impressions).
""")
c.code(r"""
users = (imp.group_by("user_code").agg(impressions=pl.len(), clicks=pl.col("n_clicks").cast(pl.Int64).sum(), exposures=pl.col("slate_len").cast(pl.Int64).sum(),
                                      active_days=pl.col("timestamp").dt.date().n_unique(), history_len=pl.col("history_len").first(),
                                      first=pl.col("timestamp").min(), last=pl.col("timestamp").max()))
summary = pd.DataFrame([describe(users["impressions"], "impressions per user"), describe(users["clicks"], "clicks per user (in window)"),
                        describe(users["active_days"], "active days per user"), describe(users["history_len"], "history length per user")])
display(save_table("usr_01_user_level_summary", summary))
ipu = (users.with_columns(bin=bucket("impressions", [1, 2, 3, 5, 10, 20], ["1", "2", "3", "4-5", "6-10", "11-20", "21+"])).group_by("bin")
       .agg(users=pl.len(), impressions=pl.col("impressions").sum(), clicks=pl.col("clicks").sum()))
ipu = ordered(ipu, "bin", ["1", "2", "3", "4-5", "6-10", "11-20", "21+"]); ipu["user_pct"] = 100 * ipu.users / ipu.users.sum(); ipu["impression_pct"] = 100 * ipu.impressions / ipu.impressions.sum()
display(save_table("usr_01_impressions_per_user_bins", ipu))
hl = (users.with_columns(bin=bucket("history_len", [0, 5, 10, 20, 50, 100, 200], ["0", "1-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201+"])).group_by("bin")
      .agg(users=pl.len(), impressions=pl.col("impressions").sum(), clicks=pl.col("clicks").sum(), exposures=pl.col("exposures").sum()))
hl = ordered(hl, "bin", ["0", "1-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201+"]); hl["user_pct"] = 100 * hl.users / hl.users.sum()
hl["clicks_per_impression"] = hl.clicks / hl.impressions; hl["ctr_pct"] = 100 * hl.clicks / hl.exposures
display(save_table("usr_01_history_length_bins", hl))
days = users.group_by("active_days").agg(users=pl.len()).sort("active_days").to_pandas(); days["user_pct"] = 100 * days.users / days.users.sum()
display(save_table("usr_01_active_days", days))
H.update(users_one_impression_pct=float(ipu.loc[ipu.bin == "1", "user_pct"].iloc[0]), users_multi_day_pct=float(days.loc[days.active_days >= 2, "user_pct"].sum()),
         median_impressions_per_user=float(users["impressions"].median()), median_history=float(users["history_len"].median()), p95_history=float(users["history_len"].quantile(0.95)),
         history_over_50_pct=100 * float((users["history_len"] > 50).mean()), empty_history_users_pct=100 * float((users["history_len"] == 0).mean()))
print(f"{H['users_one_impression_pct']:.1f}% of users appear once; {H['users_multi_day_pct']:.1f}% are active on 2+ days; median history {H['median_history']:.0f}, p95 {H['p95_history']:.0f}, "
      f"{H['history_over_50_pct']:.1f}% of users have > 50 history clicks, {H['empty_history_users_pct']:.2f}% have none")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.8))
axes[0].bar(ipu.bin.astype(str), ipu.user_pct, color=BLUE)
for xk, v in zip(ipu.bin.astype(str), ipu.user_pct): axes[0].text(xk, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8, color=MUTED)
axes[0].set_xlabel("Impressions per user in the training window"); axes[0].set_ylabel("Share of users"); axes[0].yaxis.set_major_formatter(PCT); axes[0].set_title("Most users are seen once or twice")
h = users["history_len"].to_numpy(); h = h[h <= 150]
axes[1].hist(h, bins=np.arange(0, 151, 3), color=BLUE)
axes[1].axvline(H["median_history"], color=REF, ls="--", lw=1); axes[1].text(H["median_history"] + 3, axes[1].get_ylim()[1] * 0.9, f"median = {H['median_history']:.0f}", color=MUTED, fontsize=8)
axes[1].axvline(H["p95_history"], color=REF, ls=":", lw=1); axes[1].text(H["p95_history"] + 3, axes[1].get_ylim()[1] * 0.78, f"p95 = {H['p95_history']:.0f}", color=MUTED, fontsize=8)
axes[1].set_xlabel("Supplied history length (clicks before the window; <= 150 shown)"); axes[1].set_ylabel("Users"); axes[1].yaxis.set_major_formatter(THOUSANDS)
axes[1].set_title("History length is long-tailed")
axes[2].plot(hl.bin.astype(str), hl.clicks_per_impression, marker="o", color=BLUE)
axes[2].set_ylim(1.0, None); axes[2].set_xlabel("History length"); axes[2].set_ylabel("Clicks per impression"); axes[2].set_title("Longer histories, slightly more clicks per visit")
fig.tight_layout(); save_fig(fig, "usr_01_activity_history", "User activity and history length, training split.")
""")
c.md(r"""
**Reading.** Users are light: the median user has two impressions in the 5.3-day window and 37 % appear only once. The supplied history has a
median of 11 clicks per user (19 per impression, because active users carry longer histories; p95 64, maximum 801). Beyond the shortest
histories, clicks per visit rise with history length, consistent with history length being a proxy for general engagement.

**Implications.** A history cap of 50 keeps the full history of 92 % of users; longer histories can be truncated to the most recent entries.
In-window behaviour per user is thin (median 2 impressions), so user representations must lean on the supplied history and on content, not on
in-window sequences alone.
""")

c.md(r"""
## 2. Cold-start users

Two different kinds of cold start exist here: users with **no supplied history** (2 % of impressions), and users who appear in validation or
test but **never in train** (so the model has no in-window behaviour for them, although most of them still carry a history).
""")
c.code(r"""
cold = (imp.with_columns(group=pl.when(pl.col("history_len") == 0).then(pl.lit("no history")).otherwise(pl.lit("has history")))
        .group_by("group").agg(impressions=pl.len(), users=pl.col("user_code").n_unique(), exposures=pl.col("slate_len").cast(pl.Int64).sum(),
                               clicks=pl.col("n_clicks").cast(pl.Int64).sum(), mean_slate_len=pl.col("slate_len").mean())
        .with_columns(clicks_per_impression=pl.col("clicks") / pl.col("impressions"), ctr_pct=100 * pl.col("clicks") / pl.col("exposures"),
                      impression_pct=100 * pl.col("impressions") / imp.height).sort("group"))
display(save_table("usr_02_no_history_users", cold))
H["no_history"] = cold.to_pandas().round(4).set_index("group").to_dict("index")
# new users in validation / test relative to train
tr_users = set(imp["user_code"].unique().to_list())
rows = []
for sp in ["validation", "test"]:
    d = IMP[sp].with_columns(new_user=~pl.col("user_code").is_in(list(tr_users)))
    g = (d.group_by("new_user").agg(impressions=pl.len(), users=pl.col("user_code").n_unique(), exposures=pl.col("slate_len").cast(pl.Int64).sum(),
                                   clicks=pl.col("n_clicks").cast(pl.Int64).sum(), no_history_pct=100 * (pl.col("history_len") == 0).mean(), median_history=pl.col("history_len").median())
         .with_columns(split=pl.lit(sp), ctr_pct=100 * pl.col("clicks") / pl.col("exposures"), clicks_per_impression=pl.col("clicks") / pl.col("impressions")))
    rows.append(g)
newu = pl.concat(rows).with_columns(user_share_pct=100 * pl.col("users") / pl.col("users").sum().over("split"), impression_share_pct=100 * pl.col("impressions") / pl.col("impressions").sum().over("split")).sort("split", "new_user").to_pandas()
newu["new_user"] = newu.new_user.map({True: "new vs train", False: "seen in train"})
display(save_table("usr_02_new_users_validation_test", newu))
H["new_users"] = newu.round(3).to_dict("records")
print("Users without history vs with history:", {k: round(v["clicks_per_impression"], 3) for k, v in H["no_history"].items()}, "clicks per impression")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
d = newu.pivot(index="split", columns="new_user", values="user_share_pct").reindex(["validation", "test"])
axes[0].bar(d.index, d["seen in train"], color=BLUE, label="seen in train", width=0.55)
axes[0].bar(d.index, d["new vs train"], bottom=d["seen in train"], color=ORANGE, label="new user (not in train)", width=0.55)
for i, sp in enumerate(d.index):
    axes[0].text(i, 100 - d.loc[sp, "new vs train"] / 2, f"{d.loc[sp, 'new vs train']:.1f}%", ha="center", va="center", fontsize=9, color="white")
axes[0].set_ylabel("Share of the split's users"); axes[0].yaxis.set_major_formatter(PCT); axes[0].set_ylim(0, 100); axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2); axes[0].set_title("A fifth of validation and test users never appear in train")
d2 = newu.pivot(index="split", columns="new_user", values="clicks_per_impression").reindex(["validation", "test"])
x = np.arange(2); w = 0.35
axes[1].bar(x - w / 2, d2["seen in train"], width=w, color=BLUE, label="seen in train"); axes[1].bar(x + w / 2, d2["new vs train"], width=w, color=ORANGE, label="new user")
axes[1].set_xticks(x); axes[1].set_xticklabels(d2.index); axes[1].set_ylabel("Clicks per impression"); axes[1].set_ylim(1.0, None); axes[1].legend(); axes[1].set_title("New users click somewhat less per visit")
fig.tight_layout(); save_fig(fig, "usr_02_cold_start_users", "Cold-start users in validation and test relative to the training split.")
""")
c.md(r"""
**Reading.** Users without any history are 2 % of training impressions and click exactly as much per visit as the others (1.51). The larger
cold-start group is users unseen in train: 18 % of validation users and 20 % of test users (15 % of test users are in neither train nor
validation). Almost all of them still carry a supplied history, though a shorter one (median 7 against 23 for returning users), and they click
somewhat less per visit (1.39 against 1.55).

**Implications.** The user profile must be built from the **supplied history plus content**, which exists for new users, rather than from
in-window user identity (user-id embeddings would be undefined for a fifth of test users). Users with no history need an explicit fallback
(no-history indicator, popularity and freshness priors). Report metrics separately for new and returning users at the end-to-end stage.
""")

c.md(r"""
## 3. The supplied history is a frozen snapshot: in-window clicks can extend it

The data-preparation notebook showed that a user's history string is identical across all of their impressions and across splits, including
the test release. So the test-day history of a returning user does **not** contain the clicks that user made on 9 to 14 November, although
those clicks are observed with labels and precede the test day. This section measures how much history could legitimately be added.
""")
c.code(r"""
inwin = cand.filter(pl.col("clicked") == 1).group_by("user_code").agg(train_clicks=pl.len(), last_click=pl.col("timestamp").max())
va_clicks = (pl.read_parquet(PROCESSED / "candidates_validation.parquet", columns=["user_code", "clicked"]).filter(pl.col("clicked") == 1).group_by("user_code").agg(val_clicks=pl.len()))
rows = []
for sp in ["validation", "test"]:
    u = IMP[sp].select("user_code", "history_len").unique("user_code").join(inwin, on="user_code", how="left")
    if sp == "test":
        u = u.join(va_clicks, on="user_code", how="left")
    else:
        u = u.with_columns(val_clicks=pl.lit(0))
    u = u.with_columns(added=pl.col("train_clicks").fill_null(0) + pl.col("val_clicks").fill_null(0))
    rows.append({"split": sp, "users": u.height, "users_with_in_window_clicks": int((u["added"] > 0).sum()), "users_with_in_window_clicks_pct": 100 * float((u["added"] > 0).mean()),
                 "median_added_clicks_when_any": float(u.filter(pl.col("added") > 0)["added"].median()), "mean_added_clicks_when_any": float(u.filter(pl.col("added") > 0)["added"].mean()),
                 "total_added_clicks": int(u["added"].sum()), "total_supplied_history": int(u["history_len"].cast(pl.Int64).sum()),
                 "added_as_pct_of_supplied_history": pct(int(u["added"].sum()), int(u["history_len"].cast(pl.Int64).sum())),
                 "users_with_no_history_but_in_window_clicks": int(((u["history_len"] == 0) & (u["added"] > 0)).sum())})
aug = pd.DataFrame(rows); display(save_table("usr_03_history_augmentation_potential", aug)); H["history_augmentation"] = aug.round(3).to_dict("records")
# within train: the same logic applies time-causally (clicks before the impression)
tr_seq = (imp.sort("user_code", "timestamp").with_columns(prior_clicks_in_window=(pl.col("n_clicks").cast(pl.Int64).cum_sum().over("user_code") - pl.col("n_clicks"))))
share_with_prior = 100 * float((tr_seq["prior_clicks_in_window"] > 0).mean())
H["train_impressions_with_prior_window_clicks_pct"] = share_with_prior
print(f"Training impressions preceded by at least one in-window click of the same user: {share_with_prior:.1f}%")
""")
c.md(r"""
**Reading.** 85 % of test users (82 % of validation users) have labelled clicks on 9 to 14 November that are not in their test-day history,
adding a median of 4 clicks per such user (mean 7), 23 % more history in total; about 6,000 test users with no supplied history at all would gain
one. Inside the training window the same holds time-causally: 65 % of impressions are preceded by at least one earlier in-window click of the same user.

**Implications.** Build the user profile as **supplied history + in-window clicks strictly before the impression timestamp** (for the test day:
all train and validation clicks of that user). This is leakage-free because every added click precedes the impression, and it makes the
freshest interests visible to the model. It also means the within-train "history" grows through the week while the supplied field does not.
""")

c.md(r"""
## 4. What histories contain

Histories reference articles by id, so their content is available only when the article is in the news file. The training news file is the
union of history and slate articles, so every history article has metadata; most of them, however, were never candidates in the window.
""")
c.code(r"""
t0 = time.time()
hist_long = (hist.select("user_code", "impression_uid", h=pl.col("history").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "").str.strip_prefix("N").cast(pl.Int32)))
             .unique(subset=["user_code"], keep="first").select("user_code", "h").explode("h").drop_nulls("h").rename({"h": "news_code"}))
H["history_entries_once_per_user"] = hist_long.height
hist_art = hist_long.group_by("news_code").agg(history_users=pl.len()).join(news, on="news_code", how="left")
H["history_distinct_articles"] = hist_art.height
H["history_articles_also_shown"] = int((hist_art["exposures_train"] > 0).sum())
H["history_entries_on_shown_articles_pct"] = 100 * float(hist_art.filter(pl.col("exposures_train") > 0)["history_users"].sum() / hist_long.height)
cat_mix = (hist_long.join(news.select("news_code", "category"), on="news_code", how="left").group_by("category").agg(history_entries=pl.len())
           .with_columns(history_pct=100 * pl.col("history_entries") / pl.col("history_entries").sum()).sort("history_entries", descending=True))
exp_mix = (cand.join(news.select("news_code", "category"), on="news_code", how="left").group_by("category").agg(exposures=pl.len(), clicks=pl.col("clicked").cast(pl.Int64).sum())
           .with_columns(exposure_pct=100 * pl.col("exposures") / pl.col("exposures").sum(), click_pct=100 * pl.col("clicks") / pl.col("clicks").sum()))
mix = cat_mix.join(exp_mix, on="category", how="full", coalesce=True).fill_null(0).sort("history_pct", descending=True).to_pandas()
display(save_table("usr_04_history_vs_window_category_mix", mix))
hist_cats = hist_long.join(news.select("news_code", "category"), on="news_code", how="left").group_by("user_code").agg(n_cats=pl.col("category").n_unique(), n=pl.len())
H["median_history_categories"] = float(hist_cats["n_cats"].median())
print(f"History entries (once per user): {H['history_entries_once_per_user']:,} over {H['history_distinct_articles']:,} distinct articles; "
      f"{H['history_articles_also_shown']:,} of those articles were candidates in the window, covering {H['history_entries_on_shown_articles_pct']:.1f}% of entries; "
      f"median categories per user history {H['median_history_categories']:.0f} · {time.time() - t0:.0f} s")

fig, ax = plt.subplots(figsize=(11, 4))
top = mix.head(12); x = np.arange(len(top)); w = 0.27
ax.bar(x - w, top.history_pct, width=w, color=BLUE, label="share of history entries")
ax.bar(x, top.exposure_pct, width=w, color=ORANGE, label="share of window exposures")
ax.bar(x + w, top.click_pct, width=w, color=AQUA, label="share of window clicks")
ax.set_xticks(x); ax.set_xticklabels(top.category, rotation=30, ha="right"); ax.set_ylabel("Share"); ax.yaxis.set_major_formatter(PCT); ax.legend()
ax.set_title("Histories lean more to news and sports than the slates MSN shows in the window")
fig.tight_layout(); save_fig(fig, "usr_04_history_category_mix", "Category mix of supplied histories versus window exposures and clicks, training split.")
del hist_long
""")
c.md(r"""
**Reading.** The 12.9 M history entries (counted once per user) cover 79,056 distinct articles, of which only 5,026 were ever candidates inside
the window; histories are older than the slates. Their category mix is heavier in news and sports than the slates MSN served in the window,
and a typical user's history spans 5 categories.

**Implications.** Content-based user profiles are feasible (every history article has a title and category), and they are necessary: an
id-based item model would see almost none of the history articles as candidates. Multi-category histories argue for multi-interest
representations rather than one averaged vector.
""")

c.md(r"""
## 5. Does history-category affinity predict clicks?

For each impression we rebuild the user's category profile from the history: affinity = share of past clicks in the candidate's category.
Because the profile uses only clicks before the window there is no target leakage. We bin affinity and compute candidate CTR, repeat inside
the four largest categories, and repeat on slates of length 50 to 100 to remove the slate-size confound.
""")
c.code(r"""
t0 = time.time()
cat_map = news.select("news_code", pl.col("category").cast(pl.Categorical))
affinity = (hist.lazy().select("impression_uid", h=pl.col("history").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "").str.strip_prefix("N").cast(pl.Int32)))
            .explode("h").drop_nulls("h").rename({"h": "news_code"}).join(cat_map.lazy(), on="news_code", how="left").drop_nulls("category")
            .group_by("impression_uid", "category").agg(hist_cat_count=pl.len())
            .with_columns(hist_total=pl.col("hist_cat_count").sum().over("impression_uid"))
            .with_columns(affinity=pl.col("hist_cat_count") / pl.col("hist_total")).collect(engine="streaming"))
session_hist = affinity.group_by("impression_uid").agg(session_hist_total=pl.col("hist_total").max())
print(f"impression x category affinity pairs: {affinity.height:,} · impressions with a non-empty history: {session_hist.height:,} · {time.time() - t0:.0f} s")

cand_aff = (cand.lazy().join(cat_map.lazy(), on="news_code", how="left").join(imp.lazy().select("impression_uid", "slate_len"), on="impression_uid")
            .join(affinity.lazy().select("impression_uid", "category", "affinity"), on=["impression_uid", "category"], how="left")
            .join(session_hist.lazy(), on="impression_uid", how="left").with_columns(pl.col("affinity").fill_null(0.0))
            .select("impression_uid", "slate_len", "category", "clicked", "affinity", "session_hist_total"))
AFF_EDGES, AFF_LABELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5], ["0", "0-5%", "5-10%", "10-20%", "20-30%", "30-50%", ">50%"]
def ctr_by(lf, keys):
    return lf.group_by(keys).agg(exposures=pl.len(), clicks=pl.col("clicked").cast(pl.Int64).sum()).with_columns(ctr_pct=100 * pl.col("clicks") / pl.col("exposures")).collect()
eda1 = ordered(ctr_by(cand_aff.with_columns(bucket("affinity", AFF_EDGES, AFF_LABELS).alias("affinity_bin")), ["affinity_bin"]), "affinity_bin", AFF_LABELS)
top_cats = exp_mix.sort("exposures", descending=True).head(4)["category"].to_list()
eda1_cat = ordered(ctr_by(cand_aff.filter(pl.col("category").cast(pl.Utf8).is_in(top_cats)).with_columns(bucket("affinity", AFF_EDGES, AFF_LABELS).alias("affinity_bin")), ["category", "affinity_bin"]), "affinity_bin", AFF_LABELS)
eda1_cat["category"] = eda1_cat.category.astype(str)
eda1_fixed = ordered(ctr_by(cand_aff.filter((pl.col("slate_len") >= 50) & (pl.col("slate_len") <= 100)).with_columns(bucket("affinity", AFF_EDGES, AFF_LABELS).alias("affinity_bin")), ["affinity_bin"]), "affinity_bin", AFF_LABELS)
eda1["slates"] = "all"; eda1_fixed["slates"] = "length 50-100"
display(save_table("usr_05_ctr_by_affinity", pd.concat([eda1, eda1_fixed], ignore_index=True)))
display(save_table("usr_05_ctr_by_affinity_top_categories", eda1_cat))
H.update(aff_ctr_zero=float(eda1.ctr_pct.iloc[0]), aff_ctr_top=float(eda1.ctr_pct.iloc[-1]), aff_ctr_zero_fixed=float(eda1_fixed.ctr_pct.iloc[0]), aff_ctr_top_fixed=float(eda1_fixed.ctr_pct.iloc[-1]))
print(f"CTR from affinity 0 to >50%: {H['aff_ctr_zero']:.2f}% -> {H['aff_ctr_top']:.2f}% (all slates); {H['aff_ctr_zero_fixed']:.2f}% -> {H['aff_ctr_top_fixed']:.2f}% (slates 50-100) · {time.time() - t0:.0f} s")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.9))
for ax, d, title in [(axes[0], eda1, "All slates"), (axes[1], eda1_fixed, "Slates of length 50-100 (size control)")]:
    lo, hi = wilson(d.clicks, d.exposures); x = np.arange(len(d))
    ax.bar(x, d.ctr_pct, color=BLUE, yerr=[(d.ctr_pct - 100 * lo), (100 * hi - d.ctr_pct)], capsize=3, error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(d.affinity_bin.astype(str), rotation=30); ax.set_ylabel("Candidate CTR"); ax.yaxis.set_major_formatter(PCT); ax.set_title(title)
    ax.set_xlabel("Affinity: share of the user's past clicks in the candidate's category")
piv = eda1_cat.pivot(index="affinity_bin", columns="category", values="ctr_pct").reindex(AFF_LABELS)
for cat_, col in zip(top_cats, SERIES):
    axes[2].plot(range(len(piv)), piv[cat_], marker="o", ms=3, color=col, label=cat_)
    axes[2].annotate(cat_, (len(piv) - 1, piv[cat_].iloc[-1]), textcoords="offset points", xytext=(5, 0), fontsize=8, color=MUTED, va="center")
axes[2].set_xticks(range(len(piv))); axes[2].set_xticklabels(AFF_LABELS, rotation=30); axes[2].set_ylabel("Candidate CTR"); axes[2].yaxis.set_major_formatter(PCT)
axes[2].set_title("Within the four largest categories"); axes[2].legend(loc="upper left")
fig.tight_layout(); save_fig(fig, "usr_05_affinity", "Candidate CTR rises monotonically with history-category affinity, training split.")
""")
c.md(r"""
**Reading.** CTR rises monotonically from 2.9 % at affinity 0 to 6.2 % above 50 %, a 2.1× lift. The gradient holds inside each of the four
largest categories and on slates of length 50 to 100 (1.8 % to 4.4 %), so it is neither a category-mix nor a slate-size artefact.

**Implications.** `category_affinity` is a strong, cheap, time-causal ranking feature and belongs in every ranker. The caveat is
associational: the existing recommender may already favour high-affinity items, so the lift is predictive, not causal.
""")

c.md(r"""
## 6. Novelty: does the category have to be in the history at all?

The affinity-0 bin mixes two situations: users with no history at all (cold users) and users whose history never touched the candidate's
category (unseen category). The third group has seen the category at least once.
""")
c.code(r"""
novelty = (pl.when(pl.col("session_hist_total").is_null()).then(pl.lit("cold user (no history)")).when(pl.col("affinity") == 0).then(pl.lit("unseen category")).otherwise(pl.lit("seen category")))
NOV = ["cold user (no history)", "unseen category", "seen category"]
eda2 = ordered(ctr_by(cand_aff.with_columns(novelty.alias("group")), ["group"]), "group", NOV); eda2["slates"] = "all"
eda2_fixed = ordered(ctr_by(cand_aff.filter((pl.col("slate_len") >= 50) & (pl.col("slate_len") <= 100)).with_columns(novelty.alias("group")), ["group"]), "group", NOV); eda2_fixed["slates"] = "length 50-100"
nov = pd.concat([eda2, eda2_fixed], ignore_index=True); nov["exposure_share_pct"] = 100 * nov.exposures / nov.groupby("slates").exposures.transform("sum")
display(save_table("usr_06_novelty", nov)); H["novelty"] = nov.round(3).to_dict("records")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, d, title in [(axes[0], eda2, "All slates"), (axes[1], eda2_fixed, "Slates of length 50-100")]:
    lo, hi = wilson(d.clicks, d.exposures); x = np.arange(3)
    ax.bar(x, d.ctr_pct, color=[AQUA, ORANGE, BLUE], yerr=[(d.ctr_pct - 100 * lo), (100 * hi - d.ctr_pct)], capsize=3, error_kw={"ecolor": MUTED, "lw": 0.8})
    for i, (v, n) in enumerate(zip(d.ctr_pct, d.exposures)): ax.text(i, v + 0.12, f"{v:.2f}%\n{n/1e6:.1f}M exposures", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(["cold user\n(no history)", "unseen\ncategory", "seen\ncategory"]); ax.set_ylabel("Candidate CTR"); ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, d.ctr_pct.max() * 1.4); ax.set_title(title)
fig.tight_layout(); save_fig(fig, "usr_06_novelty", "Candidates from never-clicked categories attract far fewer clicks, training split.")
""")
c.md(r"""
**Reading.** Seen-category candidates are clicked at 4.6 %, unseen-category candidates at 2.8 % (−39 % relative) and cold users at 4.1 %; the
ordering survives the size control. Unseen-category candidates are not a niche: 19.4 M of 70.3 M exposures.

**Implications.** History carries two distinct signals, intensity (affinity) and coverage (`seen_category`); keep both. Cold users behave like
average users, so the no-history fallback should be the population prior (popularity, freshness, category CTR), not a pessimistic one.
""")

c.md(r"""
## 7. Time since the previous visit: short-term carry-over versus long-term preference

For users with at least three impressions, each impression with a gap is compared with (a) the categories clicked in the user's previous
impression (short-term context), (b) the user's top history category (long-term, leak-free) and (c) the user's most-clicked category over all
their in-window impressions (long-term, retrospective). Raw bucket means mix users, so within-user demeaned values are shown as well.
""")
c.code(r"""
t0 = time.time()
GAP_ORDER = ["<1 hour", "1-6 hours", "6-24 hours", "1-3 days", ">3 days"]
gap_expr = (pl.when(pl.col("gap_hours") < 1).then(pl.lit(GAP_ORDER[0])).when(pl.col("gap_hours") < 6).then(pl.lit(GAP_ORDER[1]))
            .when(pl.col("gap_hours") < 24).then(pl.lit(GAP_ORDER[2])).when(pl.col("gap_hours") < 72).then(pl.lit(GAP_ORDER[3])).otherwise(pl.lit(GAP_ORDER[4])))
sess = (imp.select("impression_uid", "user_code", "timestamp").sort("user_code", "timestamp")
        .with_columns(gap_hours=pl.col("timestamp").diff().over("user_code").dt.total_seconds() / 3600, prev_uid=pl.col("impression_uid").shift(1).over("user_code"), n_sessions=pl.len().over("user_code"))
        .filter((pl.col("n_sessions") >= 3) & pl.col("gap_hours").is_not_null()))
clicks = (cand.lazy().filter(pl.col("clicked") == 1).join(cat_map.lazy(), on="news_code", how="left").select("impression_uid", "user_code", "category")
          .join(sess.lazy().select("impression_uid", "gap_hours", "prev_uid"), on="impression_uid", how="inner"))
prev_cat = (cand.lazy().filter(pl.col("clicked") == 1).join(cat_map.lazy(), on="news_code", how="left").select(pl.col("impression_uid").alias("prev_uid"), "category").unique().with_columns(in_prev=pl.lit(True)))
top_hist = (affinity.lazy().filter(pl.col("affinity") == pl.col("affinity").max().over("impression_uid")).unique(subset=["impression_uid"], keep="first").select("impression_uid", "category").with_columns(in_top_hist=pl.lit(True)))
user_cat = (cand.lazy().filter(pl.col("clicked") == 1).join(cat_map.lazy(), on="news_code", how="left").group_by("user_code", "category").agg(n=pl.len()))
top_lt = user_cat.filter(pl.col("n") == pl.col("n").max().over("user_code")).unique(subset=["user_code"], keep="first").select("user_code", "category").with_columns(in_top_lt=pl.lit(True))
m = (clicks.join(prev_cat, on=["prev_uid", "category"], how="left").join(top_hist, on=["impression_uid", "category"], how="left").join(top_lt, on=["user_code", "category"], how="left")
     .with_columns(pl.col("in_prev").fill_null(False), pl.col("in_top_hist").fill_null(False), pl.col("in_top_lt").fill_null(False))
     .group_by("impression_uid", "user_code", "gap_hours").agg(prev_hit=pl.col("in_prev").mean(), top_hist=pl.col("in_top_hist").mean(), top_lt=pl.col("in_top_lt").mean())
     .with_columns(gap_expr.alias("gap_group")).collect(engine="streaming"))
METRICS = ["prev_hit", "top_hist", "top_lt"]
raw = ordered(m.group_by("gap_group").agg(sessions=pl.len(), **{k: pl.col(k).mean() for k in METRICS}), "gap_group", GAP_ORDER)
dem = ordered(m.with_columns([(pl.col(k) - pl.col(k).mean().over("user_code")).alias(k) for k in METRICS]).group_by("gap_group").agg(**{k: pl.col(k).mean() for k in METRICS}), "gap_group", GAP_ORDER)
raw["view"] = "raw"; dem["view"] = "within-user demeaned"
stl = pd.concat([raw, dem], ignore_index=True); display(save_table("usr_07_short_vs_long_term_by_gap", stl))
H["short_long_term"] = {"users": int(m["user_code"].n_unique()), "sessions": m.height, "raw": raw.round(4).to_dict("records"), "demeaned": dem.round(4).to_dict("records")}
print(f"Users with >= 3 impressions: {m['user_code'].n_unique():,}; impressions with a gap: {m.height:,} · {time.time() - t0:.0f} s")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
LBL = {"prev_hit": "category repeated from the previous visit (short-term)", "top_hist": "user's top history category (long-term, leak-free)", "top_lt": "user's most-clicked category in the window (long-term)"}
for ax, d, title in [(axes[0], raw, "Raw share of the visit's clicks"), (axes[1], dem, "Within-user demeaned")]:
    for k, col in zip(METRICS, SERIES):
        ax.plot(range(5), 100 * d[k], marker="o", ms=4, color=col, label=LBL[k])
    ax.axhline(0, color=REF, lw=0.8, ls="--"); ax.set_xticks(range(5)); ax.set_xticklabels(GAP_ORDER, rotation=20); ax.set_xlabel("Time since the user's previous impression")
    ax.set_title(title)
axes[0].yaxis.set_major_formatter(PCT); axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f} pp"))
axes[0].set_ylim(0, None); axes[0].legend(loc="lower left", fontsize=7.5)
fig.tight_layout(); save_fig(fig, "usr_07_short_vs_long_term", "Short-term carry-over fades with absence; long-term alignment rises; training split, users with at least three impressions.")
del m, clicks
""")
c.md(r"""
**Reading.** About 27 to 29 % of a visit's clicks repeat a category clicked in the previous visit, almost regardless of the gap, dipping only
after 3 days; the share of clicks in the user's most-clicked category rises from 41 % to 48 % as the gap grows. Within users the movements shrink
to about ±1.5 pp (long-term alignment peaks for 6 to 24 h returns), so most of the raw gradient is composition: users who return after long gaps
have more concentrated tastes.

**Implications.** Recent clicks and the long-term profile are both informative, but the gap does not switch strongly between them. A user
representation that keeps a recent component (last visit's clicks) next to a long-term component (supplied history) is justified; the gap itself
is a weak feature on its own (see `article_temporal_analysis`).
""")

c.md(r"""
## 8. What this notebook decides

| Finding | Evidence | Decision |
|---|---|---|
| Median user has 2 impressions; history median 19, p95 110 | §1 | history cap ≈ 50 (80 % of users complete); lean on history + content, not in-window identity |
| 2 % of impressions have no history; 18–20 % of validation / test users are new | §2 | content-based profiles; explicit no-history fallback; report new vs returning users |
| Histories are frozen snapshots; most test users have labelled in-window clicks | §3 | profile = supplied history + in-window clicks before the impression time |
| History articles mostly never appear as candidates; ~7 categories per user | §4 | content (title, category, entities) is the bridge; multi-interest representation |
| Affinity lifts CTR 2.1×, robust to category and size | §5 | `category_affinity` feature |
| Unseen category −39 % CTR; cold users ≈ average | §6 | `seen_category` flag; population prior for cold users |
| Long-term alignment grows with the gap; within-user effects ±1.5 pp | §7 | recent + long-term components in the user model; gap is a weak switch |
""")
c.code(CLOSING)

write_notebook("user_analysis", c)
