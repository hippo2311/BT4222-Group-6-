**![][image1]**

**Personalised News Recommendation Using Microsoft MIND**



## **Part 1: Our Machine Learning Problem**

### **1.1 Business Objective & ML Alignment**

* **Business Pain Point:** Microsoft News needs to decide which articles should receive the most prominent positions in each user’s feed. When relevant articles are ranked too low, users click less and leave the platform sooner. This reduces engagement and page views, limiting advertising opportunities.  
* **User Pain Point:** Microsoft News users are presented with a large number of articles, making it difficult to quickly identify news that is relevant to their interests.   
* **Business Objective**: Increase the news feed’s click-through rate (CTR) by identifying articles that are relevant to each user, ranking them according to their expected relevance, and recommending the Top-K articles within a fixed-sized slate. CTR is the primary business indicator of engagement, where a higher CTR indicates that users are engaging more with the recommended content, while offline ranking metrics assess whether the model places clicked articles higher in the feed, which supports stronger platform engagement and advertising opportunities.   
* **ML Alignment:** The recommendation system first identifies candidate articles that may be relevant to the user by predicting the likelihood that the user will click each article. The candidates are then ranked according to their predicted click probabilities, and the Top-K articles are selected for recommendation. By placing more relevant articles in prominent positions with greater visibility, the model increases the likelihood of these articles attracting users’ attention, causing users to click them, which supports the business objective of increasing the feed’s click-through rate. 

### **1.2 Model Inputs & Target Outputs**

**Input Sample**

* Each input sample is a user–article pair within a specific impression, where an impression is a single instance in which a set of news articles, referred to as a slate, is displayed to a user.   
* Each impression slate is split into one sample per article shown. For example, an impression slate displaying 20 articles generates 20 samples. 

**Model Inputs**

* **User history:** history and history\_length describe the articles the user clicked before the impression. user\_interest\_cluster summarises broader interests.   
* **Candidate article:** title, category, and subcategory describe the article. Optional inputs are abstract, title\_entities, abstract\_entities, entity\_embedding, title\_length and abstract\_length.  
* **User–article match:** category\_affinity, seen\_category, multi\_interest\_match\_score and prior\_exposure\_count describe how the candidate relates to the user’s past behaviour.   
* **Article popularity and freshness:** prior\_article\_exposures, prior\_article\_clicks, smoothed\_prior\_article\_ctr, article\_age\_hours and estimated\_article\_half\_life\_hours describe activity and decay observed **before** the current impression.  
* **Slate-level comparison:** The optional personalized\_relative\_advantage compares the candidate’s match score with other articles in the same impression. graph\_article\_embedding may provide an additional article representation.

**Target output:**

* **Ground-truth label:** 1 if the article was shown and clicked, 0 if it was shown but not clicked. Articles not shown are treated as unobserved rather than negative examples.

**Model Outputs**

* **Model prediction**: An estimated click score between 0 and 1 for each candidate article. Candidates are ranked from highest to lowest score to produce the final Top-K recommendation list.

**Model Architecture:**

*Figure 1.2: Overall Recommendation System Architecture*

As shown in Figure 1.2, the recommendation system follows a two-stage architecture. We decided to adopt a full-catalogue recommendation setting rather than restricting the model to the candidate articles already pre-selected by MSN's existing recommendation system. Working from the full catalogue allows the system to decide which articles are shown in the first place, including relevant articles that the existing slates may have left out. However, scoring every article in the full catalogue for every user can be computationally expensive. Thus, the recommendation process is therefore divided into a candidate-generation stage followed by a more precise re-ranking stage.

* **Candidate Generation Model**: The catalogue is first filtered to include only articles available on or before the impression time using the available\_from timestamp (Section 3.7.1). The candidate-generation model is trained using the training data to efficiently identify articles that are potentially relevant to each user using signals such as user interests, content similarity, popularity and freshness. During validation and testing, the trained model scores the eligible article catalogue and retrieves the Top-M candidates. Its main objective is to retain as many known relevant articles as possible rather than determine their exact ranking.  
* **Re-Ranking Model**: The re-ranking model is trained using user–article pairs from the logged training impressions, where clicked articles are labelled 1 and shown-but-unclicked articles are labelled 0\. During validation and testing, the Top-M candidates retrieved by the candidate-generation model are passed to the re-ranker together with their user, article and engineered features. The model assigns each candidate an estimated click score, re-orders the Top-M candidates, and returns the highest-ranked Top-K articles as the final recommendations.  
* **Evaluation by Stage**: Both models are developed using the training split. During validation, the candidate-generation model first retrieves Top-M articles from the eligible catalogue, after which the re-ranking model re-orders these candidates and produces the final Top-K recommendations. Recall@M evaluates candidate generation, while NDCG@K, MRR and Recall@K evaluate the final ranking. Validation results are used to select the final model configuration, which is then frozen and applied to the test set for final evaluation.

**Evaluation Metrics:**

| Evaluation Metric | Formula  | What it Measures  | Relevance to Business Objective |
| :---- | ----- | :---- | :---- |
| **Recall@M** | **$\frac{#\ known\ relevant\ articles\ in\ Top-M}{#\ Total\ \ known\ relevant\ articles}$** | Measures how many of the user’s articles are successfully retrieved from the full catalogue into the Top-M candidate set | Used to evaluate candidate generation. A high Recall@M is important because any relevant article excluded at this stage cannot be recovered by the re-ranking model. |
| **Recall@K**  | **$\frac{1}{N}\sum\limits_{i=1}^{N}\frac{#\ relevant\ articles\ in\ Top-K\ for\ impression\ i\ }{#\ total\ relevant\ articles\ in\ impression\ i}$** | Measures the average proportion of known relevant articles retrieved in the final Top-K for each impression, giving equal weight to every candidate. | Used to evaluate the final recommendation output. Unlike Recall@M, which evaluates candidate generation, Recall@K measures the average proportion of known relevant articles that remain in the final Top-K after the full candidate-generation and re-ranking pipeline, giving each candidate equal weight. Unobserved articles may still lower Recall@K if they are ranked above known relevant articles and push them outside the Top-K. |
| **NDCG@K**  | **$NDCG@k\ =\frac{DCG@k}{IDCG@K}$** | Measures whether relevant articles are ranked near the top, giving greater credit to higher positions  | Suitable because users are more likely to engage with articles ranked higher in the feed. As compared to MAP@K, NDCG@K is less sensitive to unobserved articles for full-catalogue evaluation as they mainly affect the score by pushing relevant articles lower in the ranking. |
| **MRR**  | **$\frac{1}{N}\sum\limits_{i=1}^{N}\frac{1}{{rank}_{i}}$** | Measures how highly the first known relevant article is ranked. | Useful because MRR rewards models that place the first relevant article as high as possible, which can support higher engagement and CTR. However, it only considers the first relevant article and ignores other relevant articles ranked below it, which does not reflect how well the model ranks all relevant articles overall. |
| **Click Retention@K**  | **$\frac{#\ observed\ clicks\ observed\ in\ Top\ K}{#\ Total\ observed\ click}$** | Measures how many known click opportunities are preserved in the Top-K. | This metric is similar to Recall@K because both measure how many known relevant articles are captured in the Top-K, but Click Retention@K frames this from a business perspective to better show how much observed click engagement is preserved by the recommendation system. |
| **CTR Lift@K**  | **$\frac{{CTR@K}_{model}-\ {CTR@K}_{baseline}}{{CTR@K}_{baseline}}\times 100%$** | Measures the relative improvement in observed Top-K CTR compared with a baseline recommender using the same Top-K feed space. | Directly supports the business objective by comparing observed Top-K CTR against a baseline using the same feed space. A higher lift means the model retains more observed clicks in the Top-K, indicating better recommendation performance. Since unshown articles have unknown outcomes, this is an offline estimate rather than the platform’s true CTR improvement.. |
| **Engaged Impression Rate@K**  | **$\frac{#\ impression\ with\ \geq \ 1\ \ click\ in\ top\ K}{#\ Total\ impression\ }$** | Measures how often the Top-K contains at least one article that the user engages with. | This metric complements CTR by showing how consistently the recommender generates click engagement across visits. If a visit contains no relevant Top-K articles, the user is less likely to engage and may leave the platform. A higher rate therefore supports the objective of increasing overall CTR and sustaining user engagement.  |

**Note:** Unseen articles, which have unknown relevance, are treated as non-relevant during evaluation. This can bias the results by unfairly lowering the scores when such articles are ranked above observed relevant articles

## **Part 2: Datasets & Impression Data Understanding**

### **2.1 Data Overview & Sampling Scheme**

> * **Data Source:** Microsoft News Dataset (MIND), MIND-large, collected from anonymised Microsoft News (MSN) behaviour logs.  
> * We retain all valid impressions from the original MIND-large training release, then partition them chronologically into training (earliest 85%) and validation (latest 15%) sets. The separate official MIND validation release is retained as the final test set.  
> * The corresponding news subsets are then constructed based on the articles referenced in each behaviour split, either in users’ histories or impression slates. Articles referenced by behaviors\_train are included in news\_train, while articles required by behaviors\_validation are included in news\_validation. The separate official MIND validation release, together with its corresponding news data, is retained as the final test set.  
> * If negative sampling is required later to manage class imbalance during model training, it will be applied only within the training set; validation and test candidate slates will remain unchanged.  
> * Below are the summary statistics for each split and each file

**Behaviours.tsv**

| Metric / Entity | Train | Validation | Test | Description |
| :---- | :---- | :---- | :---- | :---- |
| Total impressions | 1,897,835 | 334,913 | 376,471 | Total news slate recommendation events |
| Unique Users | 667,880 | 235,963 | 255,990 | Distinct user IDs with impression logs |
| Total Clicks | 2,868,998 | 514,658 | 574,845 | Temporal coverage of user logs |
| Total Exposures | 70,314,387 | 13,192,987 | 14,085,557 | Total article-display opportunities across all slates—every candidate article shown to a user. |
| Overall ctr  | 0.040802 | 0.03901 | 0.040811 | Exposure-level click-through rate: (total clicks ÷ total exposures) |
| Start time  | 2019-11-09 00:00:00 | 2019-11-14 07:30:28 | 2019-11-15 00:00:00 | Earliest recorded impression timestamp. |
| End time | 2019-11-14 07:30:28 | 2019-11-14 23:59:59 | 2019-11-15 23:59:43 | Latest recorded impression timestamp. |
| Missing history rows | 37,121 | 8,944 | 11,270 | impressions where the user has no recorded earlier click history. These are cold-start/no-history cases and should normally be retained.  |

**News.tsv**

| Metric / Entity | Train | Validation | Test | Description  |
| :---- | :---- | :---- | :---- | :---- |
| Unique news items | 98671 | 71991 | 72023 | Distinct article identifiers represented in each split. |
| Primary categories | 18 | 17 | 17 | Count of unique top-level news categories. |
| Subcategories | 283 | 273  | 269 | Count of unique news subcategories. |
| Articles with abstract | 93411 | 68383 | 68400 | Rows containing a non-empty article abstract. |
| Abstract coverage  | 0.946692  | 0.949883  | 0.949697  | Proportion of rows with a non-empty abstract. |
| Articles with title entities | 98671 | 71991 | 72023 | Number of articles with a valid title-entity list, including empty lists (\[\]) |
| Title entity coverage | 1.0 | 1.0 | 1.0 | Proportion of articles with a valid title-entity list, including empty lists (`[]`) |
| Articles with abstract entities | 98671 | 71991 | 72023 | Number of articles with a valid abstract-entity list, including empty lists (`[]`) |
| Abstract entity coverage | 1.0 | 1.0 | 1.0 | Proportion of articles with a valid abstract-entity list, including empty lists (`[]`) |
| Mean title characters | 65.984251 | 66.234613 | 66.218375 | Average title length in characters. |
| Mean abstract characters | 216.417569 | 211.236821 | 210.412438 | Average abstract length in characters. |

### **2.2 Impression, Exposure & impression Dynamics** {#2.2-impression,-exposure-&-impression-dynamics}

This section describes the impression log at the level the model operates on: the slate, the candidate article and the click. All figures use the training split defined in Section 2.1 (1,897,835 impressions, 9 November to 14 November 2019), consistent with Section 2.3; the validation and test splits are not analysed here beyond one check that their slate structure is the same (Table 2.2.1). Two properties of the log explain most of what follows: the order of candidates within a slate was shuffled before release, and every logged impression contains at least one click. Sections 2.2.1 to 2.2.5 cover slate length, clicks per impression, position, the exposure-to-click funnel and feedback semantics. The remainder of Section 2.2 covers user sequences, article popularity and temporal patterns.

#### **2.2.1 Impression slate length** {#2.2.1-impression-slate-length}

Each row in behaviors.tsv represents one impression: the slate of candidate articles shown to a user and the corresponding click label for each article. Slate length varies considerably, from 1 to 300 articles. The median slate contains 24 candidates, while the mean is 37.0 and the interquartile range is 9-51. Most slates are relatively short: 28% contain at most 10 articles, 51.6% contain at most 25, and only 8.0% contain more than 100\. The distribution has noticeable peaks at slate lengths of 2, 4, 7, 10 and 12, suggesting that slates are generated from a small number of fixed page templates rather than by applying a continuous ranking cutoff. Slate lengths are comparable across the training, validation and test sets, with medians of 24, 26 and 23 respectively.The validation and test splits have the same structure (Table 2.2.1), so the model is evaluated on the slate sizes it is trained on.

![][image2]

*Figure 2.2.1. Slate length distribution (left, lengths up to 120; 4.6% of impressions are longer) and cumulative distribution (right). Training split, 1,897,835 impressions.*

#### **2.2.2 Clicks per impression** {#2.2.2-clicks-per-impression}

72.5% of impressions contain exactly one click, 16.3% two and 11.2% three or more (mean 1.51). No impression has zero clicks (0 of 1,897,835). Two-item slates therefore always contain exactly one click, and the log was evidently filtered to impressions that ended in a click. Because the number of clicks barely grows with slate size, item-level CTR is close to one over slate length (Table 2.2.2, Figure 2.2.2); the Spearman correlation between slate length and impression-level CTR is −0.89. The overall CTR of 4.08% is therefore a click-conditioned figure, not a platform CTR, and slate length predicts the label by construction. The number of articles displayed, listed in Section 1.2 as a possible context input, is excluded from the feature set for this reason 

Since the dataset contains only impressions with at least one click, the CTR in MIND only reflects visits where users are already engaged. Therefore, it is useful for comparing recommendations within the dataset, but the result doesn’t represent the true overall CTR of Microsoft News.

| Slate length | 2 | 5 | 10 | 25 | 50 | 100 | 200 |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| CTR (%) | 50.0 | 20.9 | 12.1 | 5.4 | 3.6 | 2.3 | 1.6 |
| Mean clicks per impression | 1.00 | 1.04 | 1.21 | 1.36 | 1.82 | 2.28 | 3.10 |
| Reference: 1 / slate length (%) | 50.0 | 20.0 | 10.0 | 4.0 | 2.0 | 1.0 | 0.5 |

*Table 2.2.2. Click-through rate and mean clicks per impression by slate length, training split.*

*![][image3]*

*Figure 2.2.2. Left: CTR by slate length (log axes) against the one-click reference curve. Right: mean clicks per impression by slate length. Lengths with at least 200 impressions, training split*

#### **2.2.3 Position in the slate** {#2.2.3-position-in-the-slate}

Pooled over all impressions, CTR falls steeply with the position of a candidate in the slate: 11.1% at position 1, 5.1% at position 10, 2.6% at position 50 and 2.0% at position 100 (Figure 2.2.3, left). This looks like position bias but is a composition effect. Short slates have a high CTR at every position and occupy only the first few positions, so pooling over slate lengths manufactures a decline: the exposures at position 1 come from slates averaging 37 items, those at position 100 from slates averaging 134\. A null model that spreads each impression's clicks uniformly over its own slate, and therefore uses only the mix of slate lengths, reproduces the observed curve within 0.27 percentage points at every position with at least 10,000 exposures.

Holding slate length fixed removes the effect entirely (Figure 2.2.3, right; Table 2.2.3). For 7-item slates CTR lies between 15.5% and 15.7% at every position, and a chi-square test of uniform click counts across positions is far from significant for all four slate lengths examined. The remaining spread is sampling noise: the standard error of a per-position CTR for 7-item slates is about 0.13 percentage points, and the observed standard deviation across positions is 0.09. Viewed by relative position (decile of the list), the same artefact appears as a U-shape, 5.1% in the first and 5.1% in the last decile against 3.5% to 4.0% in between, because two-item slates fall into both end deciles.

![][image4]

*Figure 2.2.3. Left: CTR by absolute position in the slate (solid) against the CTR predicted from the slate-length mix alone (dashed). Right: CTR by position with slate length held fixed at 7, 10, 12 and 25 items. Training split, 1,897,835 impressions.*

| Slate length | impressions | Mean CTR (%) | Min over positions (%) | Max (%) | Range (pp) | Chi-square p |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- |
| 7 | 75,854 | 15.63 | 15.46 | 15.73 | 0.27 | 0.85 |
| 10 | 54,051 | 12.11 | 11.92 | 12.30 | 0.38 | 0.82 |
| 12 | 46,516 | 9.83 | 9.62 | 10.00 | 0.38 | 0.63 |
| 25 | 24,027 | 5.44 | 5.21 | 5.76 | 0.56 | 0.80 |

*Table 2.2.3. CTR across positions with slate length held fixed. The p-value is a chi-square goodness-of-fit test of uniform click counts across positions (approximate, since clicks within a slate are not fully independent).*

Conclusion: the order of candidates in behaviors.tsv carries no information about display rank. This is consistent with the MIND documentation, which states that the order within a slate was shuffled before release (Microsoft, n.d.). No position bias can be estimated from these logs and none needs to be corrected; position is excluded as a feature. This does not contradict Section 1.1: MSN very likely has a position effect on its live pages, but the released log cannot measure it, so the case for re-ranking rests on that general finding rather than on an estimate from MIND.

#### **2.2.4 Exposure-to-click funnel** {#2.2.4-exposure-to-click-funnel}

MIND records exactly two stages of impression: an article is exposed in a slate, and it is or is not clicked. There is no dwell time, read depth, share or subscription signal, so the engagement outcomes beyond the click described in Section 1.1 are business aims rather than observable targets. Table 2.2.4 gives the funnel at four levels. At the candidate-pair level, 70,314,387 exposures produce 2,868,998 clicks (4.08%). At the impression and user levels the conversion is 100%: every impression and every user has at least one click. This is the filter described in 2.2.2, not user behaviour.

| Level | Stage | Count | Share of previous stage |
| ----- | ----- | ----- | ----- |
| Candidate pair | Exposures (article shown to a user in a slate) | 70,314,387 |   |
| Candidate pair | Clicks | 2,868,998 | 4.08% |
| Impression | Impressions logged | 1,897,835 |   |
| Impression | Impressions with at least one click | 1,897,835 | 100% |
| Article | Articles in the training split news file | 98,671 |   |
| Article | Articles shown in at least one slate | 24,641 | 25.0% |
| Article | Articles clicked at least once | 15,292 | 62.1% |
| User | Users with at least one impression | 667,880 |   |
| User | Users with at least one click | 667,880 | 100% |

*Table 2.2.4. Exposure-to-click funnel at pair,impression, article and user level, training split.*

At the article level, the 98,671 articles in the training split news file narrowed to 24,641 shown at least once (25.0%) and 15,292 clicked at least once (62.1% of those shown). The logged slates were therefore assembled by MSN's own candidate generation before the data was written, and the log carries labels only for the 24,641 articles MSN chose to show. 

This shapes both stages of the design in Section 1.2. Our candidate-generation stage retrieves Top-M articles from the full catalogue, but it can be trained and evaluated only against known clicks (Recall@M), because an article MSN never showed to a user is unobserved for that user, not rejected. Our re-ranking stage can be trained directly on the logged slates, where every candidate has an observed outcome. How clicks concentrate among the 15,292 clicked articles is examined in Section 2.2.8.

#### **2.2.5 Feedback semantics** {#2.2.5-feedback-semantics}

Labels are defined on (user, article) pairs within an impression slate. Table 2.2.4 classifies every pair the data can contain.

| (User, article) pair | Count | Meaning | Role in the model |
| :---- | ----- | ----- | ----- |
| Shown in a slate and clicked | 2,868,998 | Positive, y \= 1 | Target |
| Shown in a slate, not clicked | 67,445,389 | Explicit negative, y \= 0 | Target |
| In the user's supplied history | 12,874,380 | Past positive (clicked before the log window) | Input only, via the user profile |
| Neither shown nor in history | about 66 billion | Unobserved; label unknown, not zero | None |

*Table 2.2.5. Classification of (user, article) pairs, training split. History entries are counted once per user because the supplied history is static across a user's impressions.*

A clicked candidate is the positive class. A candidate that was exposed but not clicked is an explicit negative: the article was shown and passed over. An article in the user's supplied history is a past positive: it was clicked before the log window and entered the model only through the user profile, never as a label, because it comes with no slate to rank it against. Every other pair is unobserved; its label is unknown, not zero, because the user was never offered the article. At the article level, 74,030 of the 98,671 articles are never candidates inside the window. All of them appear in at least one user history, since the news file is the union of the articles found in histories and slates.

**Imbalance.** For training the re-ranking stage, we adopt the standard MIND scheme (Wu et al., 2019): for each clicked article, 4 unclicked articles from the same slate are drawn at random without replacement, in a single seeded draw made once before training and reused for every epoch and every model, so that all models see identical training data. Validation and test slates are kept complete, as stated in Section 2.1. Table 2.2.6 shows what the choice of K costs: with K \= 4, at least 84% of the explicit negatives never enter training, and 9.5% of positives sit in slates with fewer than 4 negatives; for those, we keep every available negative rather than duplicating one.

| Negatives per positive, K | Negatives per positive actually trained on | Explicit negatives never used (lower bound) | Positives in impressions with fewer than K negatives |
| :---- | ----- | ----- | ----- |
| 4 | 3.76 | 84% | 9.5% |
| 8 | 6.98 | 70% | 16.5% |
| 16 | 11.84 | 50% | 28.6% |
| 24 | 15.11 | 36% | 39.2% |
| 32 | 17.30 | 26% | 49.1% |
| all | 23.51 | 0% | 0.0% |

*Table 2.2.6. Effect of the sampling ratio K on the training data: single draw without replacement, training split; 'all' keeps the full slate.*

**Recommendations for sampling.** First, report the discarded share alongside every trained model. Second, run K \= 8 and K \= 16 as a sensitivity check: the unused share falls to 70% and 50% while the number of training rows stays at 2,868,998, so the extra cost is in candidates per row, not in rows. Third, the candidate-generation stage must learn to retrieve from the full catalogue, so articles outside the slate may be used as sampled training negatives for that stage (random catalogue or in-batch negatives) under the explicit assumption that most unobserved pairs would not have been clicked. That is a training device, not a label: the classification in Table 2.2.5 is unchanged, and evaluation continues to treat unobserved articles as unknown.

**Why exposed negatives are the right negatives.** They were selected by the same editorial and freshness process as the positives, so they are the hard negatives the re-ranking stage will face at serving time, and they are the only pairs with an observed label. Two evaluation views follow. Within-impression metrics (per-impression AUC, MRR and nDCG over the logged slate) use labelled candidates only and measure re-ranking performance conditional on the logged candidate slates

. Full-catalogue metrics (Recall@M for candidate generation; NDCG@K and Recall@K over the Top-K list) are the ones the business objective in Section 1.2 requires, but they must treat unobserved articles as non-relevant, so they understate performance whenever the model ranks an unobserved but relevant article above a known click. We therefore report both: the within-impression figures as measures of ranking quality on logged slates, and full-catalogue figures as measures of retrieval and ranking against observed clicks under incomplete relevance judgments. 

**Repeat exposure.** 14.2% of exposures show a user an article that was already shown to the same user earlier in the window, and 13.5% of all exposures are articles the user had passed over before. CTR is 4.33% on first exposure. On the second exposure, it is 2.62% when the article had been passed over and 5.14% when it had been clicked; on the third or later exposure, 1.85% and 3.39% (Figure 2.2.4). An explicit negative therefore, carries information beyond the impression it came from, and “previously shown and not clicked” is a usable, time-causal feature.

![][image5]

*Figure 2.2.4. CTR by exposure rank of the (user, article) pair, split by whether an earlier exposure was clicked. Training split, 1,897,835 impressions* 

**History overlap.** 0.28% of candidates and 0.58% of clicks are articles already in the user's supplied history, although 7.8% of impressions contain at least one such candidate; when re-shown, these articles are clicked at 8.4%. The supplied history is identical across all of a user's impressions (423,559 users have more than one impression, 0 of them with a changed history), so it is a static snapshot taken before the log window and does not absorb clicks made inside it. Leakage from history into the candidate set is negligible; these rows are kept and the choice documented.

#### **2.2.6 Time-of-Day Dynamics** {#2.2.6-time-of-day-dynamics}

The key question in this section is whether the timing of a recommendation affects the likelihood that a user clicks on an article. We examine temporal patterns in user activity to determine whether features such as time of day, day of week, and impression recency may be useful for the system.

![][image6]

*Figure 2.2.5. Number of recorded rows (impressions), grouped by hour*

The timeline is generated from the training split, from 9 November 2019, 12 am to 14 November 2019, 7.28 am. From the timeline, there is a significantly higher number of recorded impressions between 11 November 2019 and 14 November 2019\. 11-14 November are weekdays, although it's not possible to deduce that weekdays generate more user impressions than weekends, since there could have been something specific that happened during this week. 

![][image7]

*Figure 2.2.6. Hourly clicks per impression: raw vs within-user demeaned*

From Figure 2.2.2, since CTR is largely a function of slate length, a more representative metric of user interest is clicks per impression. To separate genuine time-of-day effects from user composition, each user’s series was centred on their own mean before hourly aggregation, so the demeaned series shows whether the same user clicks more in some hours than their own average. Single-impression users were excluded, leaving 1 653 514 impressions contributed by 423 559 users.

Both series display a daily cycle, and the demeaned curve closely tracks the raw one, with a correlation of 0.7961, indicating that the cycle is not merely a composition artifact. Within users, engagement depth is highest between roughly 2am and 5am, and lowest around 3pm. The effect is modest in magnitude \- within-user deviations rarely exceed ±0.05 clicks against a mean of 1.53, so the pattern is systematic but small relative to overall variation.

**![][image8]** 

*Figure 2.2.7. Category CTR deviation from each category’s overall CTR by hour*

The heatmap centers each row on its own mean, removing the large base-rate differences between category CTR, which range from 2.6% for travel to 6.1% for music and TV, making the time-of-day structure comparable across rows. Categories with fewer than 100 000 exposures and cells with fewer than 1000 exposures are excluded.

Two features stand out. First, the vertical banding \- all categories share the same diurnal cycle, with above-average CTR in the 7 am-10 am window and below-average CTR in the 12 am-3 pm window. This pattern suggests time-of-day rather than the changing mix of categories shown at each hour. 

Second, a few category-hour cells depart from the common pattern, music around 9 am, TV around 11 am, and weather in the afternoon. Article-level checks show these are content events rather than recurring behaviour. Each is confined to a single day, and a handful of articles account for the majority of clicks in the affected cells. 

For feature design, the stable takeaway is the shared hour-of-day pattern. 

#### **2.2.7 Impression Recency Dynamics** {#2.2.7-impression-recency-dynamics}

This section evaluates whether the time elapsed since a user’s previous impression carries a signal that justifies its inclusion as a feature. 

**Coverage**

Every user’s first impression (667 880 impressions, 35.2%) has no gap, leaving 1 229 955 impressions with a usable value. 

![][image9]

| Gap | Impressions | Candidate articles | Clicks per impression |
| :---- | :---- | :---- | :---- |
| \<1h | 198 667 | 6 819 383 | 1.450 |
| 1-6h | 313 115 | 11 737 188 | 1.548 |
| 6-24h | 406 345 | 15 464 799 | 1.607 |
| 1-3 days | 286 695 | 10 644 239 | 1.508 |
| \>3 days | 25 133 | 940 020 | 1.434 |
| Overall | 1 229 955 | 45 605 629 | 1.540 |

*Table 2.2.5. Gap coverage*

Clicks per impression peak at 6-24h and declines for near-continuation impressions under 1h and stale impressions over 3 days. 

![][image10]

 *Figure 2.2.8. Article-level CTR versus number of impressions*

However, comparing each impression to its own user’s mean clicks shrinks deviation to roughly ±0.04 clicks per impression, and the \> 3-day penalty disappears entirely. Much of the raw spread, therefore, reflects user composition. Users who return after long gaps are, on average, different users, rather than a within-user gap effect. 

Gap carries a small but consistent signal, concentrated in distinguishing immediate-continuation and stale impressions from re-engagement after 6-24h. 

Session-level features such as time of day and time since the previous session capture general user engagement rather than specific interest in an article. They can improve predictions by modelling when users are more likely to browse or click, and may reveal interactions between user activity and content type. However, they may contribute little to distinguish between candidate articles, and may mainly improve probability calibration rather than ranking quality. 

#### **2.2.8 Article Popularity** {#2.2.8-article-popularity}

| Statistic | Number of exposures | Number of clicks | CTR |
| :---- | :---- | :---- | :---- |
| Mean  | 2853.55 | 116.43 | 0.0410 |
| Standard deviation | 13645.18 | 1041.88 | 0.0785 |
| Min  | 1 | 0 | 0 |
| 25th percentile | 9 | 0 | 0 |
| Median  | 46 | 1 | 0.0178 |
| 75th percentile | 215 | 8 | 0.0517 |
| Max  | 331542 | 60638 | 1 |

*Table 2.2.6. Article popularity statistics*

The vast majority of articles are not shown that often. 75% of articles are shown less than 215 times, while 75% are clicked less than 8 times, with means of 2854 and 116, respectively, meaning a small number of highly popular articles account for a disproportionate amount of exposure and clicks. Consequently, raw article-level CTR is unreliable for articles with few exposures, and aggressive filtering based on exposure would remove a substantial portion of the catalogue.

![][image11] 

 *Figure 2.2.9. Articles sorted by number of clicks*

The top 747 articles, approximately 3.03% of the 24641 articles shown, account for 80% of clicks. This confirms the conclusion above, that most articles receive relatively little engagement, while a small group attracts the majority of user clicks. This indicates a substantial head-tail imbalance and motivates the inclusion of popularity-based features. 

![][image12]  

 *Figure 2.2.10. Number of clicks against number of exposures, each point is 1 article*

The exposure-click relationship shows a clear positive correlation. Articles with more exposures generally accumulate more clicks. This suggests that popularity is a strong candidate signal for predicting clicks. However, exposure is also influenced by the existing recommendation process, so it may reflect both article appeal and presentation bias.

![][image13]   

 *Figure 2.2.11. Article-level CTR versus number of exposures*

Article-level CTR is also highly unstable for articles with few exposures. Some low-exposure articles achieve apparently high CTRs, including values near 100%, but these estimates are based on limited observations. As exposure increases, CTR becomes more concentrated around the overall CTR of 4.08%, meaning CTR should not be used alone as a predictive feature.

These findings motivate popularity-based features such as historical exposure counts, historical click counts or smoothed CTR. Since these statistics are affected by the existing exposure policy, they must be computed using only information available before each recommendation event.

#### **2.2.9 Category exposures** {#2.2.9-category-exposures}

![][image14]

 *Figure 2.2.12. Category level CTR with number of articles shown*

From the 16 categories exposed in the slates (2 additional categories appeared only in users’ histories and were not exposed in the analysed slates), music and TV lead at 6.1%, about 2.3x travel’s 2.6%, with the remaining categories declining gradually around the 4.1% overall baseline. However, engagement does not track catalogue supply. Sports and news contribute 55% of all articles yet sit mid-pack, while music and TV have the fewest articles among the major categories but the highest CTR and by far the heaviest per-article exposure (\~8700 exposures per article vs \~2900 overall). 

![][image15]

 *Figure 2.2.13. Category level grouped into exposure buckets*

Figure 2.2.13 suggests category differences persist after exposure matching. Sports stays above the bucket average in every bucket, while travel, movies, food and drink, finance, and lifestyle stay below in every bucket, and even within the 10k+ bucket, CTR ranges from \~2.7% to \~6.2%, a spread of roughly 3.5 percentage points. Two caveats: music and TV earn their edge mainly on heavily exposed articles and slip below average in thinner buckets, so their effect is partly an exposure interaction.

The category is therefore worth keeping as a feature. It survives exposure control, spans up to \~2.3x in CTR, and is one of the few signals available for cold-start articles.

#### **2.2.10 History Behaviour** {#2.2.10-history-behaviour}

The behaviour log records, for every slate, the user’s previously clicked articles. A natural personalisation feature is: what fraction of this user’s past clicks fall in the same category as the candidate article? 

This is where the idea of affinity comes in. For each slate, we reconstruct the user’s category profile from the history column. We explode the history into article IDs, map each to its category, and count hits per category. A candidate’s affinity is the share of the user’s past clicks falling in that candidate’s category, with categories absent from the history set to 0\. For example, a user with 6 past clicks, 2 of them in news, has an affinity of 0.33 for any news candidate and 0 for a music candidate.

![][image16]

 *Figure 2.2.14. Article CTR grouped by history affinity*

Each of the 70.3M candidate exposures then carries its affinity and binary click outcome. We bin affinity (0, 0-5%, 5-10%, 10-20%, 20-30%, 30-50%, \>50%) and compute CTR per bin. CTR rises from 2.89% to 6.21%, a 2.1x increase.

The implication is that affinity is a strong, cheap-to-maintain ranking feature. However, the current recommender may already favour high-affinity items.

#### **2.2.11 Slate Novelty** {#2.2.11-slate-novelty}

Novelty splits the 0-affinity bin from the previous section into 2 different situations. A candidate article is labelled a cold user if the user has no history at all, an unseen category if the user has history but affinity is 0, and a seen category if the category appears in the history list.

![][image17]

 *Figure 2.2.15. Article CTR grouped by no history, category not present in history, and category present in history*

Pooled across all slates, seen-category candidates get 4.58% CTR, cold users 4.10%, and unseen category candidates only 2.80%. Unseen candidates are not a small niche. They are 19.4M of 70.3M exposures, while cold users contribute 1.4M. The ordering is stable under the size control, so it is not driven by small slate sizes. 

The implication is that the history articles’ categories carry 2 distinct signals: intensity (how often the user clicks a category, captured by affinity) and coverage (whether they have ever engaged with it at all). 

## **Part 3: Feature Dictionary, Descriptive Statistics and Feature Engineering** {#part-3:-feature-dictionary,-descriptive-statistics-and-feature-engineering}

This section documents the fields available in the two training Parquet files and provides descriptive statistics for their categorical, textual and sequence-based measurements. Identifiers are treated as nominal variables rather than numerical quantities. All results are computed exclusively from news\_train.parquet and behaviors\_train.parquet.

### **3.1 Feature Metadata** {#3.1-feature-metadata}

| Group | Feature | Type / unit | Definition and section |
| :---- | :---- | :---- | :---- |
| Record keys (Supplied) | session\_id | Categorical ID | Identifies an impression and groups its candidate articles.  |
| Record keys (Supplied) | user\_id | Categorical ID | Identifies the user.  |
| Record keys (Supplied) | news\_id | Categorical ID | Identifies the article.  |
| Impression context (Supplied) | timestamp | Datetime | Time of the impression; used for temporal splitting and past-only feature calculations. |
| User history (Supplied) | history | Sequence of article IDs | Articles clicked before the logged impression window. |
| User history (Engineered) | history\_length | Article count | Number of articles in history. (3.2) |
| Article content (Supplied) | title | Text | Article headline.  |
| Article content (Supplied) | abstract | Text | Article summary, when available.  |
| Article content (Supplied) | category | Categorical | Broad article topic.  |
| Article content (Supplied) | subcategory | Categorical | More specific article topic.  |
| Article content (Supplied) | title\_entities | List of entity IDs | Entities annotated in the title. |
| Article content (Supplied) | abstract\_entities | List of entity IDs | Entities annotated in the abstract. |
| Article content (Supplied) | entity\_embedding | 100-dimensional vector | Supplied vector for a matched entity; coverage is analysed in 3.5. |
| Article content (Engineered) | title\_length | Token count | Title length under the chosen tokenizer. (3.2) |
| Article content (Engineered) | abstract\_length | Token count | Abstract length under the same tokenizer. (3.2) |
| User–article match (Engineered) | category\_affinity | Proportion, 0–1 | Share of prior clicks in the candidate’s category. (2.2.10) |
| User–article match (Engineered) | seen\_category | Binary | Whether the user previously clicked an article in the candidate’s category. (2.2.11) |
| User–article match (Engineered) | multi\_interest\_match\_score | Similarity score | Best match between the candidate and the user’s interest groups. (3.7.3) |
| User–article match (Engineered) | prior\_exposure\_count | Count | Earlier exposures of this article to this user. (2.2.5) |
| Slate comparison (Engineered) | personalized\_relative\_advantage | Difference in scores | Candidate match score minus the median match score in its slate. (3.7.4) |
| Article activity (Engineered) | prior\_article\_exposures | Count | Article exposures recorded before this impression. (2.2.8) |
| Article activity (Engineered) | prior\_article\_clicks | Count | Article clicks recorded before this impression. (2.2.8) |
| Article activity (Engineered) | smoothed\_prior\_article\_ctr | Proportion | Smoothed click rate using only prior exposures and clicks (2.2.8) |
| Freshness (Engineered) | article\_age\_hours | Hours | Hours since the article’s first previously observed exposure. (3.7.5) |
| Freshness (Engineered) | estimated\_article\_half\_life\_hours | Hours | Past-only estimate of how quickly article engagement declines.  (3.7.5) |
| User Representations (Engineered) | user\_interest\_cluster | Categorical label | Cluster based on the user’s prior clicked articles. (3.7.6) |
| Article representations (Engineered) | graph\_article\_embedding | Vector | Article representation learned from the proposed graph. Graph structure is described in the current Feature Engineering section (3.7.7) |
| Target (Supplied) | click\_label | Binary | 1 for exposed and clicked; 0 for exposed and unclicked.  |
| Candidate eligibility (Engineered)  | available\_from | Datetime | Earliest observed article availability, used to determine candidate eligibility at impression time t (not model input, 3.7.1). |

### **3.2 Numeric Features** {#3.2-numeric-features}

The raw files do not contain meaningful continuous quantities; six interpretable counts are therefore measured directly from the source content for descriptive purposes: title and abstract word counts, title and abstract entity counts, history length, and impression-slate size. These measurements are not automatically treated as final engineered model inputs.

Table 3.2 reports valid count, minimum, maximum, mean, median, sample standard deviation, missing count and missing percentage. Missing text produces a missing word count; an explicit empty entity list produces a count of zero; and an empty user history is retained as a legitimate zero-length sequence. The title and abstract vocabularies contain 43,433 and 73,256 distinct case-folded tokens, respectively, under the same tokenizer used for the word-count analysis.

**![][image18]**

**Table 3.2.** Numeric descriptive measurements from the training data.

![][image19]

**Figure 3.2. Distributions of text and sequence lengths in the training data.** The displayed histograms are limited to the 99th percentile for readability, while medians and Table 2.3.9 use the full distributions.

### **3.3 Categorical Features** {#3.3-categorical-features}

The news file contains 18 nonmissing categories, 283 nonmissing subcategory labels and 306 observed category \- subcategory pairs. The three largest categories account for 67.32% of training news rows. The subcategory distribution exhibits a pronounced long tail: 117 of 283 subcategories (41.3%) contain fewer than ten articles. Consequently, aggregate patterns may be driven by frequent topics, while estimates for rare subcategories may be less stable. No label is merged or removed in this descriptive analysis.

**![][image20]**

**Table 3.3** Summary of categorical news fields.

![][image21] ![][image22]

**Table 3.3.1.** Eight most frequent categories.      **Table 3.3.2.** Eight most frequent subcategories.

![][image23]

**Figure 3.3. Category concentration and subcategory long tail in the training news data.** The category panel reports counts and shares; the rank–frequency panel uses logarithmic axes and shows how quickly subcategory support declines.

### **3.4 Text Features** {#3.4-text-features}

Using TfidfVectorizer, the top TF-IDF scores across news titles and abstracts reflects the real-world news cycle during the 5-day dataset collection window (2019-11-09 to 2019-11-14).

**![Title: Figure 4  Highest summed TF IDF scores after standard English and frequency filtering - Description: Figure 4  Highest summed TF IDF scores after standard English and frequency filtering][image24]**

**Table 3.4.1** Highest summed TF-IDF scores after standard English and frequency filtering

- National Politics & Breaking News: The prominent rank of terms like Trump, president, and state corresponds directly with major national political coverage in mid-November 2019, specifically the launch of the public U.S. House impeachment inquiry hearings  
- Sports Coverage: Terms such as game, football, season, team, and vs highlight heavy sports coverage, aligning with the active NFL and college football seasons during mid-November.

Residual Newsroom Boilerplate: High-scoring terms like new, week, year, says, and 2019 represent generic editorial boilerplate and temporal markers. While filtered through standard stop-word lists, their persistence indicates that further domain-specific custom stop-word removal (or Transformer-based contextual encoding) is beneficial before modeling.

### **3.5  Entity Features** {#3.5-entity-features}

**![][image25]**

**Table 3.5.1** Entity Embedding Structure and Coverage

The lookup contains 42,007 valid 100-dimensional vectors with no malformed rows. It covers 41,231 of the 42,628 unique entity IDs observed in training news, giving 96.72% coverage. The remaining 1,397 IDs require an unknown vector or mask.

**![Title: Figure 9  Heatmap, vector magnitude distribution and two-dimensional PCA view - Description: Figure 9  Heatmap, vector magnitude distribution and two-dimensional PCA view][image26]**

**Figure 3.5.2** Heatmap, vector magnitude distribution and two-dimensional PCA view

- Heatmap: The first 20 dimensions differ across the five sample entities, confirming distinct dense vectors.  
- Magnitude histogram: Vector lengths vary and are not fixed at one. Similarity calculations should use cosine similarity or L2-normalised vectors so larger norms do not dominate raw dot products.  
- PCA projection: The two-dimensional view shows geometric groupings and outliers in a sample of 2,000 entities. These patterns do not prove semantic clusters without entity-type labels; labelled colour coding or nearest-neighbour checks are needed before making semantic claims.

### **3.6 Missingness and Data Quality** {#3.6-missingness-and-data-quality}

A null value or blank string is classified as missing. The largest news-field missingness occurs in ‘abstract’, with 5,260 affected rows (5.33%). Tables 3.4 and 3.4.2 provide exact missingness counts for all news and behavior fields. A blank history indicates no recorded prior clicks and is therefore represented as a zero-length history rather than a missing derived numerical value.

Valid empty entity lists are reported separately from missing values. They occur in 25.67% of title-entity fields and 25.10% of abstract-entity fields. Treating \[\] as missing would therefore materially overstate the absence of supplied metadata.

![][image27]

**Table 3.6.1** Missing values in training news metadata.

#### **![][image28]**

**Table 3.6.2** Entity-field quality and valid empty annotations.

![][image29]

**Table 3.6.3** Missing values in training behavior fields.

#### **![][image30]**

**Figure 3.6.4.** Missing-field and entity-annotation coverage in the training news data. The left panel displays true missingness, whereas the right panel separately displays valid empty lists.

#### **Relation Embeddings Are Not Used** {#relation-embeddings-are-not-used}

The relation embedding file is excluded because the MIND news data provides linked entities but not the knowledge-graph edges connecting them. Relation vectors are designed for graph triplets such as:

$(DonaldTrump,\ positionheld,\ President)$

In TransE, a relation vector is meaningful when used with head and tail entities, where $h+r\approx t$. Without these triples, relation IDs cannot be connected reliably to the entities mentioned in each article. Therefore, relation vectors alone cannot provide useful article features.

Entity embeddings are sufficient and provide a more reliable extension to the text-based model. Entity embeddings can introduce structured knowledge that may not be fully captured by the article text. 

### **3.7 Feature Engineering** {#3.7-feature-engineering}

We construct features at the user, article, user–article and impression levels. For each impression, behavioural features must use only information available before its timestamp. The analyses below show whether the proposed features contain a descriptive signal; they do not establish that adding them will improve validation ranking metrics.

#### **3.7.1 Article Availability and Candidate Eligibility** {#3.7.1-article-availability-and-candidate-eligibility}

Each article is assigned an available\_from timestamp based on its earliest observed appearance in the behaviour logs or user history. For an impression at time t, only articles available on or before t are eligible for candidate generation. This prevents data leakage by ensuring that articles first observed in the future cannot be considered for earlier recommendation events. available\_from is used only for candidate filtering and is not included as a model input.

#### **3.7.2 Historical behaviour and article activity** {#3.7.2-historical-behaviour-and-article-activity}

The basic behavioural features are history\_length, category\_affinity, seen\_category, and prior\_exposure\_count. category\_affinity is the fraction of the user’s prior clicked articles in the candidate’s category; seen\_category records whether that category appears at least once. prior\_exposure\_count counts earlier exposures of the same article to the same user. For article popularity, prior\_article\_exposures, prior\_article\_clicks and smoothed\_prior\_article\_ctr will be calculated from events before the impression being scored. Smoothing is needed because CTR estimates from articles with very few exposures are unstable.

Sections **2.2.5, 2.2.8, 2.2.10 and 2.2.11** provide the relevant exploratory results. In particular, category affinity and prior exposure distinguish groups with different observed click rates. These findings motivate the features, although the observed rates also reflect which articles the existing system chose to show.

#### **3.7.3 Multi-interest semantic match** {#3.7.3-multi-interest-semantic-match}

A single averaged history vector may hide a user’s distinct interests. To construct multi\_interest\_match\_score, we represent article titles using TF–IDF, reduce the vectors to 64 dimensions with Truncated SVD, and normalise them. We group each user’s previously clicked articles by category and retain up to the three categories with the most clicks. Averaging the article vectors within each group produces an interest vector. The candidate’s score is its **highest cosine similarity** to these interest vectors. Users with no prior clicks receive a separate no-history indicator and fallback value.

In the training-data analysis, clicked articles have higher median scores than unclicked articles for users with one, two, and three interest groups: **0.097 versus 0.068**, **0.188 versus 0.140**, and **0.309 versus 0.248**, respectively. Scores can rise mechanically when the maximum is taken over more groups, so comparisons should be made *within* each group-count category. This is evidence of an association with clicks, not yet evidence of a gain in model performance.

![][image31]

Figure 3.7.1. Median semantic-match scores for clicked and unclicked candidates, grouped by number of user interests; training data

#### **3.7.4 Personalized relative advantage** {#3.7.4-personalized-relative-advantage}

personalized\_relative\_advantage asks whether a candidate matches the user better than the other articles in the **same impression**. We calculate it as:

$relative\ advantag{e}_{u,a,s}=\ multiinterest\ matc{h}_{u,a}\ -\ media{n}_{j\ ∈\ s}(multiinterest\ matc{h}_{u,j})$

Subtract the median Feature 1 score of the slate from each article’s Feature 1 score. For example, if an article scores 0.31 and the slate median is 0.25, its relative advantage is \+0.06.

Among clicked exposures, **59.3%** have a positive relative advantage, compared with **49.0%** of unclicked exposures. This describes how the two groups differ; it is not the probability of a click given a positive value. Subtracting the same median from every candidate cannot change their within-slate order **when this is the only score used**. We include it to test whether a model can use slate context together with other features.

![][image32]

Figure 3.7.2. Share of clicked and unclicked exposures with positive relative advantage; training data

#### **3.7.5 Article freshness and estimated half-life** {#3.7.5-article-freshness-and-estimated-half-life}

Articles have different observed lifespans, measured from first to last recorded exposure. Median lifespan ranges from 3.8 hours for Video to 28.4 hours for Lifestyle. Because an article’s last exposure is unknown at prediction time, observed lifespan is used only for analysis.

![][image33]Figure 3.7.5a. Observed lifespan by category for articles exposed in the training data. Dots show medians and lines show the middle 50% of articles. Lifespan is measured from recorded exposures, not publication time.

We instead calculate article\_age\_hours from the first recorded exposure and estimate estimated\_article\_half\_life\_hours using **only earlier exposures and clicks**. The estimate compares click rates in the first six observed hours and the six most recent completed hours. It is available only when both windows have enough exposures and the rate has declined. Article age is available for all **70.31 million** candidate exposures; half-life is estimable for **20.30%**, with a median of **21.0 hours** among those estimates. Missing estimates require a separate indicator in the model.

**![][image34]**Figure 3.7.5b. Distribution of article age, distribution of supported past-only half-life estimates, and reasons an estimate is unavailable. Counts are at the candidate-exposure level in the training data.

We then tested a **six-hour grace rule**: flag an article when its age exceeds its estimated half-life by more than six hours; retain articles without an estimate. On the logged slates, this flags **7.74% of candidates**, containing **5.13% of observed clicks**.

![][image35]

Figure 3.7.5c. Shares of candidates and their observed clicks flagged by the rule.

These are retrospective counts from existing slates. They motivate testing the rule, but cannot establish how clicks or CTR would change if articles were removed.

#### **3.7.6 Clustered user-interest profile** {#3.7.6-clustered-user-interest-profile}

To capture a user's broader preferences in a compact form, we construct one interest-profile vector from the articles that the user has previously clicked. We mean-pool the available entity embeddings of those articles and standardise the resulting user vectors before clustering.

We apply K-Means to training-user vectors and select K \= 5\. The elbow analysis shows the largest reductions in the within-cluster sum of squares between K \= 2 and K \= 5, after which the incremental improvement becomes smaller. Five clusters, therefore, provide a practical balance between separation, model complexity, and interpretability. Users with no prior clicks or too little usable entity history are assigned a separate cold-start label rather than being forced into a behavioural cluster.

![][image36]

Figure 3.7.6. Elbow Method for user-interest cluster

For prediction, the vector and cluster assignment must be derived only from clicks available before the impression timestamp. The cluster label is a candidate user-level feature; its contribution will be assessed by comparing the same ranking model with and without it on identical validation slates.

User-interest clusters are fitted only on training users. Each user vector is constructed solely from the supplied history field, which contains clicks before the logged impression window. The fitted scaler and K-means model are then frozen and applied to validation and test users using their own supplied histories. Clicks recorded in the behaviour logs are not used to create cluster assignments.

#### **3.7.7 Constructed heterogeneous knowledge graph** {#3.7.7-constructed-heterogeneous-knowledge-graph}

Although MIND provides entity annotations, it does not provide directly usable knowledge-graph triples linking the supplied relation embeddings to article entities. We therefore construct a heterogeneous knowledge graph from the training news metadata and behaviour logs. The graph contains user, article, entity, category and subcategory nodes.

The constructed graph contains five relation families: 

(i) article-to-entity mentions parsed from title\_entities and abstract\_entities;

(ii) article-to-category links

(iii) article-to-subcategory links

(iv) user-to-article click links from historical behaviour

(v) entity-to-entity co-occurrence links. 

A co-occurrence edge is retained only when the two entities appear together in at least two distinct articles, with edge weight equal to the number of shared articles. Reverse relations are added mechanically so that information can propagate in both directions. For each prediction event, user-to-article links are restricted to clicks observed before that event to avoid temporal leakage.

The current construction contains 238,939 article-to-entity edges, 197,342 article-to-topic edges, 80,774 entity co-occurrence edges and 6.57 million historical user-to-article click edges. Under this construction, 87.9% of articles have at least one matched entity. These counts demonstrate substantial relational coverage, but they do not by themselves demonstrate an improvement in recommendation quality.

A relational graph convolutional network can be trained on this graph to produce a graph\_article\_embedding for each article. Its value will be established only through a controlled comparison of the same baseline ranker with and without graph embeddings, evaluated on identical validation slates using AUC, MRR and NDCG@K. The clustering and graph features are complementary: clustering provides a compact user representation, while the graph provides an article representation informed by entities, topics and historical impressions.

## **Conclusion** {#conclusion}

The MIND dataset provides sufficient behavioural, textual and entity information to support a two-stage personalised news recommendation system. Our proposed architecture first retrieves a Top-M candidate set from the time-eligible article catalogue, before applying a more detailed re-ranking model to produce the final Top-K recommendations. The exploratory analysis motivates the use of user-interest, content-similarity, popularity, freshness and historical-impression features, while also highlighting important limitations such as click-conditioned logs and unobserved article relevance. The next stage of the project will train and compare candidate-generation and re-ranking approaches, evaluate candidate retrieval using Recall@M, and evaluate the final recommendations using ranking and business-oriented metrics on the validation and test sets.