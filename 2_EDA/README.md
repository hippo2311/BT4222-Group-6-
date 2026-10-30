# 2. Exploratory data analysis

BT4222 Group 6 · final project · task 3. Consolidates the team's interim EDA (report sections 2.2 and Part 3) into three notebooks that run end to end on the processed
files written by `1_Data_Preparation/1. data_preparation.ipynb`. These notebooks were originally developed with MIND-large; rerun and verify numerical findings on MIND-small. Descriptive figures use the **training split**;
validation and test enter only where cold start is measured.

| Notebook | Scope | Figures |
|---|---|---|
| `impression_analysis.ipynb` | slate length, clicks per impression, position, funnel and feedback semantics, negative sampling, repeat exposure, slate composition | `figures/imp_*.png` |
| `user_analysis.ipynb` | user activity, history length, cold-start users, static histories and in-window augmentation, history content, affinity, novelty, short- vs long-term interest | `figures/usr_*.png` |
| `article_temporal_analysis.ipynb` | catalogue structure, popularity, category CTR, freshness and time-causal popularity, cold-start articles, time of day, recency gap | `figures/art_*.png` |

Each notebook is a sequence of short cells in the style of the previous group's `2_eda.ipynb`: a numbered heading that states the
finding and its model implications, one computation that prints its numbers, one plot. Every number quoted in the report is printed
in the notebook next to the plot. Run order: data preparation first, then any of the three notebooks (1 to 3 minutes each; 16 GB RAM).

## Findings that drive the feature and model design

| Finding | Where | Consequence |
|---|---|---|
| Every impression has a click; CTR ≈ 1 / slate length (Spearman −0.89) | imp §1 | `slate_len` excluded; evaluate per impression; CTR is a within-log yardstick only |
| Position carries no display-rank information (null model within 0.3 pp; chi-square p 0.63–0.85) | imp §2 | `position` excluded; no position-bias correction |
| Labels exist only for shown pairs; 75 % of the news file is never a candidate | imp §3 | unobserved ≠ negative; retrieval scored against known clicks; full-catalogue metrics are lower bounds |
| 23.5 negatives per positive; K = 4 discards ≥ 84 % | imp §4 | single seeded draw, K = 4; report discarded share; K = 8 / 16 sensitivity |
| Re-shown passed-over articles are clicked less (2.6 % vs 4.3 %) | imp §5 | `prior_exposure_count` with previous outcome, time-causal |
| Diversity and stand-out effects vanish at fixed slate length | imp §6 | slate-context features low priority |
| Median user has 2 impressions; history median 11 per user, p95 64 | usr §1 | history cap 50; rely on history + content |
| 18–20 % of validation / test users are new; 2 % of impressions have no history | usr §2 | content-based profiles; explicit no-history fallback |
| Histories are frozen snapshots; 85 % of test users have labelled in-window clicks (+23 % history) | usr §3 | profile = supplied history + in-window clicks before the impression |
| Affinity lifts CTR 2.1× (2.9 % → 6.2 %), robust to category and slate size | usr §5 | `category_affinity` |
| Unseen category −39 % CTR; cold users ≈ average | usr §6 | `seen_category`; population prior for cold users |
| 3 % of shown articles take 80 % of clicks (Gini 0.96); raw CTR unstable below 100 exposures | art §2 | prior exposures / clicks / smoothed CTR, never whole-window CTR |
| Category CTR spread (2.6–6.1 %) survives exposure control | art §3 | `category` feature, compact encoding |
| CTR decays with age (5.9 % → 2.5 % after two days); 1,000+ prior clicks → 6.2 % CTR | art §4 | `article_age_hours` + censoring flag; hourly time-causal popularity, modelled jointly with age |
| Observed lifespan (first to last exposure) runs from 4 h (video) to 28 h (lifestyle) over all shown articles, 6 h to 39 h for articles with ≥ 10 exposures | art §4 | category-specific decay; the past-only half-life estimate and six-hour grace rule live in Bryan's `Interim_2.3b_Bryan` notebook (sections 6i, 6j), not here |
| 70 % of test-day exposures (76 % of clicks) are on articles unseen in train | art §5 | content-based candidate generation; popularity must update through the test day |
| Time of day moves volume (×2 weekday vs weekend), not engagement (±0.07 clicks) | art §6 | coarse day-part feature, calibration not ranking |
| Gap effect small, partly composition | art §7 | recency feature with first-visit flag, low priority |

## Conventions

* Plain pandas / matplotlib / seaborn code; polars is used only to aggregate the 70 M-row candidate table, which pandas cannot hold.
* One colour order (blue, orange, aqua, yellow), one y-axis per panel. Re-run the notebook rather than editing a figure.
* Lifespan statistics use every shown article, the definition of the interim report (section 3.7.5); the ≥ 10-exposure cohort is reported alongside because single-exposure articles have a zero span.
* Data vocabulary: *impression* = one slate shown to one user (one row of behaviors.tsv); *exposure* = one (impression, candidate)
  pair; *slate length* = tokens in the impression string; *position* = the shuffled order of those tokens; *history* = the user's
  pre-window clicks, identical across all of the user's impressions and across splits.
