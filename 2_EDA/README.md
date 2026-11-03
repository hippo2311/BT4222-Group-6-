# 2. Exploratory data analysis

BT4222 Group 6 · final project · task 3. Three notebooks on the **MIND-small** processed files written by
`1_Data_Preparation/1. data_preparation.ipynb` (train = 9 to 12 Nov 2019, 95,071 impressions; validation = 13 Nov; test = 14 Nov).
Descriptive figures use the **training split**; validation and test enter only where cold start is measured.

| Notebook | Scope | Figures |
|---|---|---|
| `impression_analysis.ipynb` | slate length, clicks per impression, position, funnel and feedback semantics, negative sampling, repeat exposure, slate composition | `figures/imp_*.png` |
| `user_analysis.ipynb` | user activity, history length, cold-start users, static histories and in-window augmentation, history content, affinity, novelty, short- vs long-term interest | `figures/usr_*.png` |
| `article_temporal_analysis.ipynb` | catalogue structure, popularity, category CTR, freshness and time-causal popularity, cold-start articles, time of day, recency gap | `figures/art_*.png` |

Each notebook follows the style of the previous group's `2_eda.ipynb`: pandas only, a numbered heading that states the finding and its
model implications, one computation cell with numbered steps that prints its numbers and draws its plot. Every number quoted in the report
is printed in the notebook next to the plot. Each notebook runs in well under a minute.

## Findings that drive the feature and model design (MIND-small, training split)

| Finding | Where | Consequence |
|---|---|---|
| Every impression has a click; CTR ≈ 1 / slate length (Spearman −0.90) | imp §1 | `slate_len` excluded; evaluate per impression; CTR is a within-log yardstick only |
| Position carries no display-rank information (null model within 0.3 pp; chi-square p 0.17–0.63) | imp §2 | `position` excluded; no position-bias correction |
| Labels exist only for shown pairs; 69 % of the news file is never a candidate | imp §3 | unobserved ≠ negative; retrieval scored against known clicks; full-catalogue metrics are lower bounds |
| 22.9 negatives per positive; K = 4 discards ≥ 84 % | imp §4 | single seeded draw, K = 4; report discarded share; K = 8 / 16 sensitivity |
| Re-shown passed-over articles are clicked less (2.6 % vs 4.4 %) | imp §5 | `prior_exposure_count` with previous outcome, time-causal |
| Diversity and stand-out effects vanish at fixed slate length | imp §6 | slate-context features low priority |
| Median user has 2 impressions; history median 12 per user, p95 66 | usr §1 | history cap 50; rely on history + content |
| 28–29 % of validation / test users are new; 1.8 % of impressions have no history | usr §2 | content-based profiles; explicit no-history fallback |
| Histories are frozen snapshots; 80 % of test users have labelled in-window clicks (+19 % history) | usr §3 | profile = supplied history + in-window clicks before the impression |
| Affinity lifts CTR 2× (3.0 % → 6.1 %), robust to category and slate size | usr §5 | `category_affinity` |
| Unseen category −36 % CTR; cold users ≈ average | usr §6 | `seen_category`; population prior for cold users |
| 4 % of shown articles take 80 % of clicks (Gini 0.95); raw CTR unstable below 100 exposures | art §2 | prior exposures / clicks / smoothed CTR, never whole-window CTR |
| Category CTR spread (2.7–6.6 %) survives exposure control | art §3 | `category` feature, compact encoding |
| CTR decays with age (≈6 % in the first hours → ≈3 % after two days); 100+ prior clicks → 6.4 % CTR, 1,000+ → 16 % | art §4 | `article_age_hours` + censoring flag; hourly time-causal popularity, modelled jointly with age |
| 77 % of test-day exposures (85 % of clicks) are on articles unseen in train | art §5 | content-based candidate generation; popularity must update through the test day |
| Time of day moves volume (×2 weekday vs weekend), not engagement (±0.07 clicks) | art §6 | coarse day-part feature, calibration not ranking |
| Gap effect small, partly composition | art §7 | recency feature with first-visit flag, low priority |

## Conventions

* pandas / matplotlib / seaborn only; no shared helper module. Re-run the notebook rather than editing a figure.
* Lifespan statistics use every shown article (the definition of the interim report, section 3.7.5); the ≥ 10-exposure cohort is
  reported alongside because single-exposure articles have a zero span. The past-only half-life estimate and six-hour grace rule live
  in stage 4 (`4_01_build_ranking_features.ipynb`), not here.
* Data vocabulary: *impression* = one slate shown to one user (one row of behaviors.tsv); *exposure* = one (impression, candidate)
  pair; *slate length* = tokens in the impression string; *position* = the shuffled order of those tokens; *history* = the user's
  pre-window clicks, identical across all of the user's impressions and across splits.
