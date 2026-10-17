# Article/user representations and ranking features: implementation plan

Prepared for Bryan's contribution to BT4222 Group 6, 16 October 2026.

Implementation update: stages 3/4 are short notebooks with visible settings and results, backed by shared implementations in `src/newsrec/`. The original artifact/temporal contracts below still apply; no external JSON settings file is required.

## 1. Purpose and scope

Build the shared representation and feature layer for the project's **full-catalogue, two-stage news recommender**: retrieve Top-M time-eligible articles, then rank those candidates into Top-K recommendations. The work must support both evaluation on complete logged slates and end-to-end evaluation on retrieved candidates. These are different candidate distributions and must remain separate in outputs and reporting.

Your core deliverables are:

1. `3_Representations/3_01_news_embeddings.ipynb`: reproducible title/abstract representations, mappings, coverage and cost audits.
2. `3_Representations/3_02_entity_embeddings.ipynb`: validated entity lookup and pooled article entity vectors, with explicit missingness.
3. `3_Representations/3_03_user_profiles.ipynb`: supplied-history profiles and profiles updated using earlier observed clicks; long-term, recent and multi-interest views.
4. `4_Feature_Engineering/4_01_build_ranking_features.ipynb`: one feature implementation usable for both logged candidates and retrieved Top-M candidates.
5. Shared configuration, reusable functions, schemas, manifests, checks and handoff documentation. These are necessary dependencies for the later model notebooks, even though they were absent from the short task description.

This plan does not implement the baseline models, retrieval algorithms, MLP, NRMS, graph learner or clustering model. It specifies what those owners will receive and how their outputs reconnect to this layer. The interim report is project evidence and a record of proposed methods; its descriptive findings are not proof that every proposed feature improves ranking.

## 2. What the existing project establishes

The repository currently implements acquisition, preparation and EDA. Representation and model folders are still planned. Raw splits, entity vectors and `1_Data_Preparation/processed/` are absent from this checkout; numerical facts below come from committed reports and tables, not a fresh data run. Timing and memory figures later in this document are estimates until measured.

### 2.1 Inputs to reuse

| Input | Existing contract | Planned use |
|---|---|---|
| `processed/news_catalogue.parquet` | 104,151 union articles; `news_id`, `news_code`, text, categories, entity annotations, split flags and retrospective article facts | Static content and ID mappings; explicitly select safe columns |
| `processed/impressions_<split>.parquet` | `impression_uid`, user IDs/codes, timestamp, supplied history and summary counts | Impression requests, histories and chronological replay |
| `processed/candidates_<split>.parquet` | `impression_uid`, `user_code`, timestamp, `position`, `news_code`, `clicked` | Historical exposure/click events and labelled training/evaluation pairs |
| `processed/users.parquet` | Mapping for 750,434 union users | Identity joins; user codes are not continuous predictors |
| `train/entity_embedding.vec` | Training-release 100-dimensional entity vectors | Canonical entity coordinate system |
| `test/entity_embedding.vec` | Dev-release vectors, used by preparation's test coverage audit | Compatibility audit; do not automatically mix with train vectors |

Paths above are relative to `1_Data_Preparation/` except the two raw vector paths. Reuse the existing local/Colab root-discovery convention rather than hardcoding a personal directory.

### 2.2 Split and evaluation contract

| Split | Impressions | Candidate exposures | Time window, using the dataset log clock |
|---|---:|---:|---|
| Train | 1,897,835 | 70,314,387 | 9 Nov 2019 to 14 Nov 2019 07:30:28 |
| Validation | 334,913 | 13,192,987 | 14 Nov 2019 07:30:28 to 23:59:59 |
| Test | 376,471 | 14,085,557 | 15 Nov 2019 |

The official MIND-large dev release is this project's labelled test set. Keep the saved 85/15 training-release partition; do not rerun acquisition simply to make a new split. Train and validation share one boundary second. `session_id` restarts in the dev release, so use `impression_uid` and never join across splits on `session_id` alone.

### 2.3 Findings that determine priorities

| Verified finding | Consequence for this work |
|---|---|
| Median supplied history is 11 articles per training user; p95 is 64; median user has two logged impressions | Use content profiles, a default sequence cap of 50 and explicit truncation audits; avoid relying on learned user identity |
| Supplied histories are static across impressions and splits | Append observed clicks strictly before each request; do not mistake the supplied field for a current profile |
| 18.4% of validation users and 19.8% of test users are absent from train | Distinguish new identity from no history; a new user may still have a useful supplied history |
| 5.33% of training articles lack abstracts; titles are present | Title fallback is mandatory; missing abstract must not drop a candidate |
| Training entity-ID coverage is 41,231 / 42,628 = 96.72% | Audit article/profile coverage separately; entity-ID coverage is not article coverage |
| About 70% of test exposures are on articles never exposed in train | Transform unseen article content and update behavioural state causally; zero historical popularity is valid |
| Category affinity has a strong descriptive association with clicks | Make category affinity and seen-category core features |
| Raw article CTR is unstable at low exposure counts | Use past-only exposure/click support, smoothing and low-support indicators |
| Every logged impression contains a click; slate length strongly determines logged CTR | Exclude `slate_len` and current click-count summaries from model inputs |
| Candidate order was shuffled; pooled position effects vanish after controlling slate length | Exclude `position`; no position-bias correction based on this field |
| Slate diversity/stand-out effects weaken at fixed slate size | Treat slate context as an optional ablation, after the core features |

Use current EDA manifests/tables for numerical claims. Some narrative cells have stale history summaries, and the interim report's selected-category lifespan range is not an all-category range. Do not copy those summaries into final results without checking their source tables.

## 3. Rules to settle before implementation

### 3.1 One temporal replay policy

Use **prequential evaluation with frozen learned parameters** as the primary protocol: score requests at time `t`, then reveal their logged feedback for use at later times. Model weights, text transforms, scalers, clusters and graph parameters remain frozen during validation/test. Observed historical state may change.

- Training features at `t` use only exposure/click events at timestamps `< t`.
- Validation starts from training events strictly before each request. Earlier validation feedback may update state after scoring.
- Test starts from all earlier train and validation events; earlier test feedback may update state after scoring.
- All impressions with the same timestamp read the same pre-timestamp state. Apply all their updates only after all have been scored. Session-ID sorting does not establish causal order inside a second.
- At the shared train/validation boundary second, training events at that same second cannot enter validation features for that second.
- Use logged feedback, not invented feedback on newly recommended articles. This is replay under MSN's historical exposure policy, not an interactive simulation of the proposed recommender.
- Also expose a frozen-state evaluation mode for sensitivity: no feedback from the evaluated split is consumed. Record the mode in every result so teams do not compare unlike experiments.

For simplicity, assume logged feedback becomes available after its timestamp batch. MIND does not provide a click-delay model; document this assumption. Test labels must never tune feature settings, model parameters or eligibility rules.

Reusable replay structure:

```text
for each timestamp t in merged chronological events:
    admit content observable at t under the availability policy
    build profiles and historical statistics from events with time < t
    score every request at t with frozen transforms and parameters
    expose the batch's logged feedback and update historical state
```

A single timestamp-aware state builder must supply training feature files and serving-time feature calls. Store a state cutoff and protocol version with every cache.

### 3.2 Article availability is a proxy, not publication time

The report proposes `available_from` from the earliest observed slate/history appearance. The dataset provides no reliable publication timestamps and no timestamps for individual supplied-history clicks. Implement the following observable convention:

1. An article first seen in a supplied history becomes known at that user's first observed impression containing it. Do not invent the click time or backdate all validation/test histories to training start.
2. An article first appearing in a logged slate becomes known at that slate's timestamp. At a timestamp boundary, admit all newly observable content before scoring requests at that timestamp.
3. `available_from` is the minimum of these observable discovery times. Eligibility is `available_from <= t`. Precomputing discovery timestamps is acceptable only if the query still enforces that cutoff and does not use later facts.
4. Encode static union metadata in advance for caching, but never equate presence in the union catalogue or a split news file with eligibility at an earlier time.
5. Do not expire articles using their last observed exposure or remove previously clicked articles by default. Repeated clicks exist and the existing data contract keeps them.

This is an **observed-availability catalogue**, not the true MSN inventory. It still depends on logged exposures and current supplied histories; it cannot prove real publication-time eligibility. Report catalogue size over time, clicked-item eligibility coverage, discovery source and newly admitted articles. A stricter `< t` discovery sensitivity can show the cost of first-exposure uncertainty; retain excluded positives in the overall Recall@M denominator and report eligibility loss separately.

Keep `available_from` separate from `first_observed_exposure`: a history-only article may be known but have no observed exposure. Availability is a filter, not an input feature. History discovery must not increment article exposure counts.

### 3.3 Fitted transforms and safe input columns

- Fit TF-IDF, SVD, vocabulary, categorical encoders and numerical preprocessing on training data only. Transform validation/test with frozen artifacts; reserve unknown-category and unknown-token codes.
- Standard supervised train-only text fitting may use the full training news subset. It is not a claim that the encoder itself was historically trained before every training request. If a strict rolling deployment simulation is needed, fit at earlier cutoffs and label it a separate experiment.
- All label-derived training aggregates, including smoothing priors, must still be point-in-time; a full-training CTR cannot be used on early training rows.
- Never expose catalogue `exposures_<split>`, `clicks_<split>`, `last_exposure_<split>`, split membership flags, current `n_clicks`, `n_negatives`, or labels through a broad feature join. Whole-window totals remain unsafe even after an article's first exposure.
- Treat keys as identity fields. In particular, union-fitted integer user codes must not become numeric features or silently create trainable representations for unseen users.

## 4. `3_01_news_embeddings.ipynb`: text representations

### 4.1 First implementation

Use a CPU-friendly TF-IDF + 64-dimensional Truncated SVD baseline, matching the report's multi-interest method. A larger contextual encoder is a later optional experiment after coverage and runtime are known.

1. Validate the catalogue key and compare duplicate IDs across raw split metadata. Report text/category/entity conflicts; use deterministic source precedence already used in preparation, rather than silently changing article content.
2. Create an explicit dense `article_row` mapping. `news_code` is an identifier with gaps, not a NumPy row index. Reserve padding/unknown rows separately for sequence models.
3. Apply minimal deterministic text cleaning: normalise whitespace and Unicode, preserve names and numbers, and use a recorded tokenisation configuration. Do not add a hand-built stop list just because words look generic in EDA; test its effect if attempted.
4. Fit title TF-IDF and SVD on unique training article IDs. Starting configuration: word 1-2 grams, maximum 50,000 terms, `min_df=2`, fixed seed, SVD dimension 64. These are proposed defaults, not previously validated settings.
5. Freeze the transform and produce title vectors for all catalogue articles. Transform abstracts through the same title-fitted TF-IDF/SVD space so pooled title/abstract vectors have compatible coordinates. Report abstract OOV/zero-vector rates; a separately fitted abstract model is an optional alternative, with its own space and profiles.
6. L2-normalise valid title and abstract vectors. Save both independently. For a combined text view, use `normalise(0.7 * title + 0.3 * abstract)` when both are valid, otherwise the valid field alone. These weights are validation-selectable defaults.
7. Record zero vectors, OOV rates, abstract absence, finite-value checks and norm distributions. Titles being present does not guarantee a nonzero TF-IDF vector. A zero content vector routes to the remaining entity/popularity/freshness fallback, with a validity flag.
8. Keep `title64` as the report-comparable multi-interest representation and `text64` as a separate combined-content experiment. An experiment must record which view it uses.

### 4.2 NRMS handoff

NRMS needs token sequences and history masks; an average SVD vector is not an NRMS implementation. Export train-fitted title vocabulary and padded token IDs, proposed title length 30 tokens, plus masks and OOV/truncation summaries. Let the NRMS owner own its trainable news/user attention encoders and checkpoint-specific article vectors. If contextual/subword tokenisation is later selected, version it separately rather than changing these arrays in place.

### 4.3 Outputs

Under `3_Representations/artifacts/`: `article_index.parquet`, `title64.npy`, `abstract64.npy`, `text64.npy`, title token IDs/masks, fitted TF-IDF/SVD/vocabulary artifacts and `news_representation_manifest.json`. Store a representation row's validity/missingness in the index or a keyed quality table. Every array must have documented shape, dtype, reserved rows, content hash and row-map hash.

## 5. `3_02_entity_embeddings.ipynb`: entity and article vectors

1. Parse `title_entities` and `abstract_entities` as JSON, preserving `WikidataId`. Distinguish valid empty annotations, missing fields, invalid JSON and unmatched IDs.
2. Validate canonical training lookup rows: 100 numeric dimensions, finite values, duplicate IDs, zero norms and file hash. The report records 42,007 valid lookup vectors; verify it when data is available.
3. Use the training vector table as the default space for all splits. The preparation test coverage number uses the dev lookup; recompute test coverage using the canonical train table as well.
4. Audit overlap between train/dev vector tables for shared IDs and coordinate equality. Equal dimension does not imply aligned spaces. Do not splice dev-only vectors into the training space without verified common coordinates and an explicit experiment policy; unmatched test entities can safely remain masked.
5. Deduplicate IDs within each article across title and abstract. First version: equal-weight average of valid, individually L2-normalised entity vectors, followed by article-level L2 normalisation. Keep valid counts and valid/annotated fraction. Mention-frequency or confidence weighting is optional and must be versioned.
6. Use a zero article-entity vector plus `has_entity_vector=0` if nothing matches. Mask unavailable vectors before pooling and cosine computation; never treat a zero-vector cosine as observed evidence. Do not impute text vectors into the entity space.
7. Export the parsed article-entity links for the graph owner, preserving the distinction between an annotation link and a matched embedding.
8. Exclude `relation_embedding.vec` from this feature layer, consistent with the report: the corresponding head/relation/tail triples are unavailable.

Outputs: canonical entity ID/row map and matrix; `article_entity100.npy`; `article_entity_links.parquet`; entity quality table and manifest. Provide separate annotation coverage, unique-entity lookup coverage, article-vector coverage and history/profile coverage. A high ID match rate can coexist with articles having no annotated entities.

## 6. `3_03_user_profiles.ipynb`: profiles at request time

### 6.1 Canonical history state

Maintain two sources without losing their provenance:

- **Supplied history:** seed once from the user's snapshot when that user is first observed. These clicks have no individual timestamps. Preserve the file's sequence, but verify/document its chronology before treating the last IDs as the newest.
- **Observed click events:** append all clicked articles from earlier timestamp batches. Keep distinct genuine repeat-click events, even when their article ID appears in supplied history. Do not re-seed the snapshot at each impression or append an event twice.

All clicks from one timestamp batch are unordered relative to one another. Use deterministic ID ordering for storage where needed, and record that this is not click chronology. Never include another click from the current impression in the candidate's profile.

Keep full historical counts for affinity, repeat behaviour and diagnostics. Use a capped representation/sequence history of 50 articles for affordable pooling and attention. When snapshot order is unverified, use its full mean for the long-term vector, and a deterministic capped subset for sequences with the limitation recorded; do not label that subset as recent. Record history length before cap, retained length, matched-vector fraction and truncation rate. Validate the cap of 50; consider 20/100 only as sensitivity settings.

### 6.2 Profiles to provide

| Profile | Definition | Consumer |
|---|---|---|
| Supplied-history text profile | Normalised mean of valid supplied-history article vectors | Fixed content baseline; stable long-term anchor |
| Causal text profile | Normalised mean over valid capped supplied + earlier observed clicks | Content retrieval and text cosine features |
| Recent text profile | Earlier observed clicks only; initial default last 10 events within 24 hours | Short-term match; missing if no such events |
| Entity profile | Normalised mean over valid article entity vectors in the causal history | Entity cosine features |
| Category/subcategory distribution | Counts and shares over all supplied + earlier clicked articles | Affinity, novelty, diversity and multi-interest grouping |
| Up to three category interest vectors | Group capped history by category; choose three groups by full causal category counts, then mean-pool valid title64 vectors in each selected group | `multi_interest_match_score`; optional multi-vector retrieval |
| Supplied-history-only entity profile | Full supplied-history entity mean, independent of replay updates | Training/frozen user clustering in report §3.7.6 |
| History sequence | Article rows, title token references, masks and observed-time information where available | NRMS and future sequence models |

For the three-interest representation, omit groups with no valid vectors and expose both selected-category count and valid interest-vector count. Break category-count ties deterministically. Record the capped-history choice as an explicit engineering refinement of the report's uncapped definition; keep an uncapped reproduction option for a fair comparison. Never mix unaligned text/entity coordinates.

A proposed combined long/recent profile is optional: blend normalised long-term and recent vectors with a validation-selected weight and renormalise. First deliver both components independently so the downstream model can test their value.

### 6.3 Cold-start behaviour

Distinguish `new_user_vs_train`, `no_supplied_history`, `no_causal_click_history`, `no_valid_text_history`, `no_valid_entity_history` and `no_recent_clicks`. Most of these are different conditions. The first is an evaluation cohort flag, not automatically a model input.

If text history is unavailable, use population popularity/freshness/category-prior fallback through the baseline/retrieval layer. If only entity history is absent, retain text personalisation. Empty profiles use zero vectors plus masks; omit invalid interests from maxima, and use a documented score fallback of zero when no valid interests exist. This value is distinguishable from an observed zero similarity by the validity flags.

Persist immutable supplied profiles, compact history/state references and replay checkpoints. Compute changing vectors on demand or cache by `(user_code, profile_state_version, representation_version)`. A final user vector keyed only by user ID must never be reused for earlier requests.

## 7. `4_01_build_ranking_features.ipynb`: shared feature layer

Keep notebooks focused on settings, execution and result inspection. Define shared representation, temporal-state, feature and validation logic in `src/newsrec/` and import it from the notebooks. Notebooks exchange saved data artifacts; the shared implementation prevents repeated definitions from drifting.

### 7.1 Feature dictionary and priority

| Priority/group | Features | Definition and handling |
|---|---|---|
| Core: content | Category, subcategory, title/abstract word counts, abstract missingness, entity counts/coverage | Train-fitted categorical maps with unknown code; static metadata only |
| Core: history | `history_length`, capped length, valid text/entity history fractions, no-history/missing-vector flags | State before request; retain raw counts for audit, use `log1p` where appropriate |
| Core: user-article match | `category_affinity`, subcategory affinity, `seen_category`, text/entity/long-term/recent cosine | Category affinity = prior clicked articles in candidate category / all prior clicked articles; zero plus no-history flag for empty denominator; mask invalid cosine |
| Core: repeated exposure | `prior_exposure_count`, `prior_nonclick_count`, `prior_pair_click_count`, time since last exposure, previously clicked | Same user/article events strictly before request; supplied history supports a separate ever-clicked flag but supplies no exposure/nonclick count or time |
| Core: article activity | `prior_article_exposures`, `prior_article_clicks`, `smoothed_prior_article_ctr`, low-support flag | All historical exposures, never sampled training rows; counts before request |
| Core: freshness | `article_age_hours`, no-prior-exposure flag, left-censoring flag | Age since first previously observed exposure; zero/flag on a first exposure, missing/flag for never-exposed eligible articles |
| Report extension: multi-interest | `multi_interest_match_score`, valid group count, profile-missing flag | Maximum cosine to valid category-interest title64 vectors; zero + mask when no valid group |
| Report extension: half-life | `estimated_article_half_life_hours`, availability flag/reason, support counts, proposed `stale_after_grace` | Completed past windows only; never substitute observed last-exposure lifespan |
| Optional: recent activity | Previous completed 1/6/24-hour clicks/exposures and smoothed CTR | Windows end at the completed-hour cutoff; choose a small set after cost audit |
| Optional: request context | Coarse day part, weekday, time since previous impression, first-visit flag | Dataset log clock; missing interval with first-visit flag; low priority because shared request features alone do not order candidates |
| Optional: slate context | `personalized_relative_advantage`, context-valid flag | Candidate match minus median candidate match in the actual candidate set being ranked |
| Later integration | `user_interest_cluster`, `graph_article_embedding`, graph/profile coverage masks | Versioned optional artifacts, with leakage and unseen-node handling checked before inclusion |

For reproducibility, the feature schema must list each column's dtype, scope, formula, missing value, dependency, cutoff rule, preprocessing, feature-group tag and permitted consumer. Labels and diagnostic cohort fields must live outside the model-input allowlist.

### 7.2 Historical article CTR

Let `E_a(t)` and `C_a(t)` be exposures and clicks on article `a` strictly before `t`. Use:

```text
p_global(t) = (all_prior_clicks(t) + 1) / (all_prior_exposures(t) + 2)
smoothed_prior_article_ctr(a,t) = (C_a(t) + lambda * p_global(t)) / (E_a(t) + lambda)
```

The fixed Beta(1,1) startup prior avoids using future training labels; it is 0.5 before any events and should be flagged as startup state. Proposed `lambda=20`, selected against validation alternatives if necessary. Include support counts because the same smoothed value can have very different reliability. A category-specific expanding prior is a later alternative, not required for the first deliverable.

Compute behavioural state from the complete historical exposure stream before negative sampling. Never estimate CTR from sampled negatives. Frozen-state evaluation freezes the corresponding aggregate state too.

### 7.3 Freshness and half-life

Maintain separate `available_from` and `first_observed_exposure`. For articles exposed before `t`, compute age from the latter. Mark articles first exposed at the start of observation as left-censored: an observed age of one hour does not imply publication one hour ago. History-only known articles have unknown exposure age. At a first exposure, use the documented zero-age proxy and no-prior-exposure flag without counting that current exposure in popularity.

Implement the report's half-life as a supported optional feature:

1. Build per-article hourly exposure/click state from the complete stream. At time `t`, use only completed log-clock hours ending no later than `floor_hour(t)`.
2. Reference window: first six complete hourly bins beginning at the first full hour on/after first observed exposure. Recent window: latest six completed bins. Require disjoint, complete windows and exposure support in both; partial/overlapping windows cannot establish decay.
3. Compare their smoothed CTRs using a fixed, versioned smoothing rule. When `0 < r_recent < r_early`, estimate `h = ln(2) * delta_hours / ln(r_early / r_recent)`, with `delta_hours` the window-centre separation. Otherwise mark unavailable. Record the assumption of exponential CTR decline and that exposure-policy shifts can affect it.
4. Proposed minimum support is 100 exposures per window. The report does not state its exact thresholds, smoothing or bin alignment, and its original `Interim_2.3b_Bryan` notebook is absent. Recover it if available; otherwise label the implementation above a new specified version and do not promise to reproduce its percentages.
5. Emit missing reasons: too young, insufficient support, overlapping windows, no decline, or invalid estimate. Any numerical model imputation must preserve the missingness flag; unavailable estimates must not be interpreted as expired articles.
6. Calculate `stale_after_grace = age > h + 6` only for supported estimates. First use it as a feature/diagnostic. Test hard filtering only in a separate validation ablation and report positive eligibility loss as well as retrieval gain/cost.

The report's 20.30% estimability, 21-hour median estimate and 7.74% flagged-candidate / 5.13% flagged-click shares are historical descriptive results, not acceptance targets or guaranteed reproduction results. Category decay can later use supported estimates from earlier completed data with a fallback; do not turn whole-window observed lifespans into prediction features.

### 7.4 Slate context and deployment consistency

For candidate set `S` and multi-interest score `m(u,a,t)`:

```text
personalized_relative_advantage(u,a,S,t) = m(u,a,t) - median(m(u,j,t) for j in S)
```

For logged-slate experiments, use the complete original slate, not its sampled training subset. For end-to-end inference, use the actual deduplicated retrieved Top-M set, not the original MSN slate. Do not use click labels to construct the median or context validity. When profile scores are unavailable, use the documented fallback and mark context invalid.

Subtracting a common median does not change ranking under that score alone. Its value can arise only through model interactions or calibration and must be demonstrated. The logged/retrieved context shift may make it unreliable: keep the main deployable feature set free of slate context until tested, and report a separate context-enabled variant. Do not introduce retrieval scores as ranker features unless an equivalent, leakage-safe training value has been defined.

### 7.5 Training rows and inference interface

Use a common API contract:

```python
build_features(requests, candidate_pairs, state_before_t, representation_bundle,
               feature_config, candidate_source)
# returns feature rows aligned to (impression_uid, news_code)
```

`candidate_source` identifies `logged` or `retrieved` for provenance, not as a model feature. Retrieved inference pairs have no required label or position. Validate unique pair keys; unexpected duplicates must fail or be resolved explicitly, not cause a many-to-many join.

For training sampling, follow the report: one deterministic draw of up to four exposed unclicked candidates per positive, without replacement within each draw; retain all available negatives when fewer exist. Persist positive-group membership and a canonical union pair table so repeated use across positive groups does not silently multiply pointwise rows. Training objective owners must declare whether they consume grouped positive/negative examples or unique pointwise pairs. Record per-group selections, reuse multiplicity, seed, discarded-negative share and sampling weights if used. Consider K=8/16 only as a shared sensitivity experiment.

Validation and test logged slates remain complete. Inference on retrieved candidates must not manufacture negative labels for unshown articles. Any optional population calibration of sampled-training scores requires unsampled validation evidence and remains limited by click-conditioned logs.

## 8. Artifact and handoff contracts

All names in this section are proposed additions. Use compact Parquet for keyed tables, float32 NumPy/memory-mapped arrays for dense vectors, and JSON for schemas/manifests. Large reproducible artifacts should be gitignored when implementation begins; commit notebooks, code, configuration and compact audit results.

| Artifact | Key/shape | Downstream dependency |
|---|---|---|
| `3_Representations/artifacts/article_index.parquet` | Unique `news_code` -> dense `article_row`; static metadata and validity | Every model and array consumer |
| `title64.npy`, `abstract64.npy`, `text64.npy` | Article rows x 64, float32 | Content baseline, retrieval, MLP match features |
| `article_entity100.npy`, entity lookup/mappings | Article rows x 100; entity rows x 100 | Entity matching, clustering and graph owners |
| Title token IDs/masks | Article rows x fixed token length | NRMS owner |
| `article_availability.parquet` | `news_code`, discovery timestamp/source, first exposure and censoring provenance | All candidate generators and end-to-end evaluation |
| Supplied profile matrices and index | User identity -> row; separate text/entity validity | Baseline; frozen clustering input |
| Replay state/checkpoints and request profile references | Cutoff, user, state/representation version | Causal features, retrieval and NRMS histories |
| `4_Feature_Engineering/artifacts/ranking_features_<split>/` | Partitioned keyed scalar features; vector row references | Simple ranker, MLP; validation/test logged-slate comparisons |
| `training_negative_groups.parquet` | Positive pair -> sampled negative IDs, seed/version | Shared ranking training examples |
| `feature_schema.json`, fitted preprocessing, configuration and manifests | Feature allowlist, group tags, hashes and temporal policy | Reproducibility, ablation and pipeline compatibility |
| `results/representation_coverage.csv` | Split, representation, cohort, denominator, valid/fallback count/rate | Model coverage and cold-start reporting |
| `results/representation_cost.csv` | Stage, input size, hardware, wall time, peak RAM, artifact size | Cost assessment and reproduction |
| `results/feature_quality.csv`, `results/feature_cost.csv` | Feature/group, split, missingness, validity, timing | Ranker handoff and ablation planning |

Features may be materialised for sampled training pairs and full validation/test pairs, but historical state must always come from the complete earlier logs. Store reusable article/user vectors once; never repeat their full dimensions on every candidate row. Keep labels in a separate keyed table or an explicitly excluded target column.

A manifest must record source hashes, saved split hashes, Git revision, configuration, library versions, seed, representation dimensions, ID-map hashes, cutoff policy, evaluation update mode, feature-schema version and build time. A consumer should reject mismatched maps/configuration rather than silently concatenate incompatible arrays.

### 8.1 Specific downstream handoffs

| Later part | What your work provides | What its owner must keep consistent |
|---|---|---|
| 5. Popularity/freshness baseline | Past-only counts, smoothed CTR, age, eligibility, no-history fallback | Same replay policy and complete validation requests |
| 5. Content-based baseline | Article vectors, causal/supplied profiles, masks and cosine routine | Representation version, profile definition, fallback and eligibility |
| 5. Simple logged-slate ranker | Core scalar feature table, schema, training sample groups, preprocessing | Shared sampled train data; complete validation slates |
| 6. Candidate generation | Row maps, vectors, multi-interest profiles and availability filter | Same eligible catalogue; deduplication; deterministic ties; new article index updates |
| 7. MLP re-ranking | Core/multi-interest feature groups, vector references and optional context | Same labelled impressions and preprocessing; retrieved inference has no labels |
| 7. NRMS re-ranking | Title tokens, capped causal history article rows and masks | Correct attention model; no current clicks; report any extra scalar-feature extension |
| 8. User-interest clustering | Supplied-history-only entity profiles, train-user membership, coverage | Fit scaler/K-means only on training users; frozen assignments from supplied histories, as in report §3.7.6; K=5 is the report's proposed choice |
| 8. Graph embeddings | Parsed article/entity/topic links, temporal click-event interface, common row map | No validation/test future clicks; no full-training click graph used to featurise earlier training events; explicit unseen-article fallback |
| 8. Ablation | Named feature groups, flags, fixed row keys and sampling version | Same ranker/data/seed/candidate sets; remove groups without dropping uncovered examples |
| 9. End-to-end evaluation | Shared feature API, replay state, eligibility and cost/coverage reports | Generate features on retrieved Top-M; separate logged-slate and full-catalogue results |

For graph features, require time-respecting training snapshots or another explicitly justified leakage-safe design. A single graph using the last training click can leak that click into earlier training rows even if validation/test edges were excluded. For new articles, use an agreed inductive metadata encoder or content fallback; never use a future-trained embedding without declaring it.

## 9. Coverage, computation and verification

### 9.1 Coverage measurements

Report each representation by train/validation/test and by cohort: all catalogue articles, eligible articles, shown candidate exposures, clicked exposures, history entries, no-history users, new-vs-returning users and articles never exposed in train. Include both unique-ID and event-weighted denominators.

Separate text availability, nonzero text vectors, entity annotation availability, entity lookup coverage, valid article entity vectors, valid user profiles and fallback usage. Keep overall model/request coverage at 100% through explicit fallback where possible; do not obtain better metrics by dropping difficult examples. Report actual scoring failures separately from fallback coverage.

For retrieval, distinguish eligible-known-click coverage, vector-index coverage, retrieved known-click recall and fallback usage. Report Recall@M against all known clicked items for the request, with optional eligible-only recall beside it. Otherwise an aggressive age filter could appear to improve retrieval simply by excluding difficult positives from the denominator.

### 9.2 Cost plan

Use the existing 16 GB RAM target. Estimates below use decimal MB/GB and exclude array headers, reserved rows and intermediate allocations.

| Object | Approximate storage/work | Implementation implication |
|---|---|---|
| One 104,151 x 64 float32 article matrix | 26.7 MB | Three text views are about 80 MB |
| One 104,151 x 100 float32 entity matrix | 41.7 MB | Keep entity and text spaces separate |
| One 750,434 x 64 float32 user matrix | 192.1 MB | Multiple profiles and versions add up; do not persist every request vector blindly |
| Three 64D interests for every union user | 576.3 MB plus masks | Cache active users or compute on demand |
| 750,434 x 50 int32 history-row array | 150.1 MB plus masks | Store article references, not copied embeddings |
| All 97.59M exposures x 30 float32 scalar features | 11.7 GB before keys, overhead and temporary joins | Partition by time/split, select columns, use lazy/streaming processing |
| 97.59M exposures x 164 copied vector coordinates | About 64.0 GB before overhead | Never expand text/entity matrices across all exposure rows |
| Exhaustive 64D scoring over the union catalogue | About 6.7M multiply-accumulate terms per request before filtering | Retrieval owner must batch, cache or index; do not assume exhaustive scoring of every request is cheap |

TF-IDF stays sparse; never densify a 104k x 50k vocabulary matrix. SVD produces the compact dense views. Profile construction should use incremental sums/counts and bounded histories where possible; avoid repeated history-by-candidate joins. Per-request exact multi-interest scoring is at most three 64D dot products per candidate, not a fresh clustering operation.

Benchmark a reproducible small chronological slice, then a complete time partition, then the full run. The test slice must carry its preceding state or be explicitly a startup experiment. Measure wall time, peak resident memory, rows/articles per second, artifact size, cache-hit rate and feature latency per request/candidate batch. Include CPU/GPU, RAM, batch sizes and cold/warm cache state. Use a proposed 12 GB peak-RAM engineering budget to leave headroom within 16 GB; reduce batch size or partition if exceeded. No runtime target should be presented as achieved before measurement.

### 9.3 Meaningful acceptance checks

These are implementation checks to add with the notebooks; this planning task has not executed the data pipeline.

- **Identity and joins:** one article-row mapping; all input pairs retained exactly once; all referenced arrays/maps agree; no missing join keys; padding/unknown cannot collide with a real article.
- **Vector integrity:** expected dimensions/dtypes; finite coordinates; valid nonzero vectors have unit norm within tolerance; zero/unmatched cases have masks; no implicit space mixing.
- **Profile causality:** supplied history seeded once; current clicks excluded; same-second impressions cannot affect one another; repeated genuine click events remain distinguishable from duplicate ingestion.
- **Future-mutation check:** adding/changing future labels cannot alter earlier profiles, popularity or feature values with frozen representation artifacts. Changing the current label cannot change that row's features.
- **Boundary cases:** shared train/validation second, first-ever exposure, history-only article, first-ever user, new user with history, missing abstract, no annotated entities and unmatched entity IDs.
- **Popularity correctness:** compare incremental counts and windows with a tiny hand-calculated event stream; sampled training tables cannot affect historical counts; global and article smoothing priors have valid cutoffs.
- **Freshness correctness:** future last-exposure changes cannot change past age; unsupported/overlapping/nondeclining half-life cases remain unavailable; no nonpositive or nonfinite estimates reach a model.
- **Slate context:** invariant to candidate ordering; no labels used; full logged versus retrieved contexts are correctly identified; singleton/invalid-profile behaviour is defined.
- **Serving parity:** identical request/candidate/state input gives identical features through offline and inference entry points; retrieved pairs need no target fields.
- **Reproducibility and cost:** fixed seeds and manifests reproduce small-slice outputs; full chronological processing fits the memory budget; no dense all-catalogue-by-all-user allocation.

## 10. Execution order and review gates

| Stage | Work | Completion evidence |
|---|---|---|
| A. Establish inputs and contracts | Obtain agreed raw/processed files; verify hashes/schema; define timestamp, availability, sampling and feature contracts | Configuration and compact audit; no resplitting; missing inputs fail clearly |
| B. Text and entity representations | Build the two article notebooks, maps, tokens and validity flags | Matrices, frozen transforms, coverage/cost reports; meaningful sample nearest-neighbour checks without claiming clusters from plots |
| C. User state and profiles | Implement replay, supplied/recent/causal profiles, three-interest vectors, masks and fallback interfaces | Hand-calculated temporal examples pass; profile coverage/truncation report; fixed clustering input available |
| D. Core features | Historical counts, affinity, repeat behaviour, smoothed CTR, age and censoring | Sampled training and full validation features; schema; future-mutation and join checks pass |
| E. Report extensions | Multi-interest features, supported half-life/grace diagnostics, optional relative advantage | Separate named groups, support/missingness report, logged/retrieved context parity checks |
| F. Handoff and integration | Demonstrate one complete logged request and one retrieved Top-M request through the API; attach README/schema/manifests | Model owners can load artifacts without rebuilding definitions; no missing-label dependency at inference |
| G. Frozen final build | Select configuration on validation; rebuild compatible artifacts; process test chronologically once settings are frozen | Reproduction instructions, cost/coverage CSVs, feature group versions and final handoff |

Deliver the core layer before waiting for clustering or graph work. Those are optional additions with explicit masks and compatible IDs. If time is limited, prioritise correct replay, coverage and core features over expensive representation experiments or low-priority slate context.

## 11. What was missing from the original task list

The short task description needs these explicit additions to be usable later:

- Article eligibility and its observation-time limitations, separate from freshness.
- Dense article indexing, train-fitted transforms, token sequences and masks for NRMS.
- Temporal state updates across all splits, including tied timestamps and frozen-state sensitivity.
- Separate supplied, recent, causal and clustering profiles; no-history and missing-vector fallbacks.
- A versioned definition for half-life and its missing-support reasons; grace filtering treated as an experiment.
- Whole-log historical aggregation before training-negative sampling, shared sample groups and declared objectives.
- The same feature API for logged slates and retrieved candidates, with explicit slate-context distribution shift.
- Feature-group tags and compatible graph/cluster interfaces for controlled ablation.
- Coverage denominators, memory budgeting, reproducible artifacts and handoff acceptance checks.

Success means later notebooks can reuse the same eligible catalogue, representation version, user state and feature definitions without guessing or rebuilding them. Ranking improvements remain for the model/evaluation stages to establish on the shared validation set; offline logged results do not establish live CTR uplift.

## 12. Evidence and reference locations

Local project sources used for this plan:

- [Interim report](</Users/bryan03/Downloads/Interim Report (1).pdf>), especially §§1.2, 2.1–2.2 and 3.5–3.7.7. §3.7.3 defines title TF-IDF/SVD64 and up to three category interests; §3.7.4 defines median-relative advantage; §3.7.5 motivates supported past-only half-life; §3.7.6 fixes supplied-history clustering inputs; §3.7.7 describes the proposed graph.
- [Project README](README.md), [acquisition README](0_Data_Acquisition/README.md), [preparation README](1_Data_Preparation/README.md) and [EDA README](2_EDA/README.md): saved split, processed-file contracts and downstream project layout.
- [Preparation audit](1_Data_Preparation/data_quality_report.csv) and [preparation implementation](notebook_builders/build_prep.py): entity coverage, stable keys, catalogue construction and retained missingness.
- [User manifest](2_EDA/manifest_user_analysis.json), [article manifest](2_EDA/manifest_article_temporal_analysis.json), [impression manifest](2_EDA/manifest_impression_analysis.json) and [EDA tables](2_EDA/tables/): current numerical findings. These are the preferred local source for figures reused in the final report.

Unresolved inputs for implementation: the actual ignored data/vector files; the original half-life notebook/configuration if exact reproduction is required; and confirmed supplied-history ordering before using its tail as a recency signal. They do not prevent this plan from specifying conservative defaults and explicit fallback behaviour.
