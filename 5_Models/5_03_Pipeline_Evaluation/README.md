# 5.03 Pipeline evaluation

These are the **final notebooks** of the project. Each one is self-contained: it trains (or loads)
its own models from the prepared data and writes its metrics to `results/`. No other notebook has to
be run first.

## Run order

Run them top to bottom (**Runtime > Run all** in Colab), in this order:

| # | Notebook | What it shows | Trains |
|---|---|---|---|
| 1 | [1_RR_only/PE_RR_logged_slates.ipynb](1_RR_only/PE_RR_logged_slates.ipynb) | **Reranker only.** Popularity and content-similarity rules vs the RR4 attention reranker, reordering the exact slates MIND logged | RR4 reranker |
| 2 | [2_CG_RR_E2E/PE_popularity_reranker_baseline_pipeline.ipynb](2_CG_RR_E2E/PE_popularity_reranker_baseline_pipeline.ipynb) | **End to end, baseline.** Combined generator (200 popularity + 250 content) → popularity reranker → final Top-K | Nothing |
| 3 | [2_CG_RR_E2E/PE_content_reranker_baseline_pipeline.ipynb](2_CG_RR_E2E/PE_content_reranker_baseline_pipeline.ipynb) | **End to end, baseline.** Same generator → content-similarity reranker → final Top-K | Nothing |
| 4 | [2_CG_RR_E2E/PE_attention_reranker_niche_pipeline.ipynb](2_CG_RR_E2E/PE_attention_reranker_niche_pipeline.ipynb) | **End to end, final system.** Niche two-tower generator (200 popularity + 250 niche) → RR4 attention reranker → final Top-K | Generator and RR4 reranker |
| 5 | [results.ipynb](results.ipynb) | Compares notebooks 2–4 on the test split, then shows notebook 1's table | Nothing |

Each notebook starts with one **Configure** cell (section 0), the only cell meant to be edited:

| Setting | Default | Meaning |
|---|---|---|
| `RUN_ENV` | `"auto"` | `"colab"`, `"local"`, or `"auto"` to detect Colab |
| `COLAB_PROJECT_PATH` | `"/content/drive/MyDrive/BT4222-Group-6-"` | Shared project folder on Colab (also found one folder down in My Drive) |
| `LOCAL_PROJECT_PATH` | `None` | Project folder when running locally; `None` = the repository the notebook is in |
| `COLAB_DATA_PATH` | `None` | Prepared data on Colab (`1_Data_Preparation/processed`, `3_Representations/artifacts`, `4_Feature_Engineering/artifacts/prequential-all`); metrics go to `<data path>/results`. `None` = the project folder |
| `LOCAL_DATA_PATH` | `"/Users/numkoos/Downloads/v3-mindsmall"` | The same for local runs; change it to your own data folder |
| `MODEL_PATH` | `None` | Where trained models are saved and loaded (notebooks 1 and 4); `None` = `<data path>/5_Models/5_03_Pipeline_Evaluation/artifacts/<model>` |
| `TARGET_SPLIT` | `"test"` | Split to evaluate (`"test"` or `"validation"`) |
| `TRAIN_IMPRESSIONS` | `None` | Reranker training impressions; `None` = whole training set (notebooks 1 and 4) |
| `EVAL_IMPRESSIONS` | `None` | Target impressions evaluated; `None` = every impression (notebooks 2–4) |
| `REUSE_SAVED_MODELS` | `True` | Load a saved model whose settings match instead of retraining (notebooks 1 and 4) |

The defaults run the whole dataset. A number such as `5_000` for `TRAIN_IMPRESSIONS` or
`EVAL_IMPRESSIONS` gives a quicker run on a random sample.

### On Google Colab

The notebooks detect Colab automatically. They mount Google Drive and look for the
`BT4222-Group-6-` folder (code and data together) in My Drive, either directly or one folder down.

1. In Google Drive, right-click the shared folder, then choose **Organise > Add shortcut > My Drive**.
   You need **Editor** access, because the notebooks save their models and results into that folder.
2. Open a notebook from Drive with **Open with > Google Colaboratory**.
3. For notebooks 1 and 4, choose **Runtime > Change runtime type > T4 GPU**.
4. Run **Runtime > Run all**.

### Locally

Run from the repository with the prepared data restored into it (see the
[project README](../../README.md)), or point `NEWSREC_DATA_ROOT` at a separate data folder.

## Two kinds of evaluation

- **Reranker only** (`1_RR_only`): the candidates are the articles MIND actually showed, so every
  candidate has a click label and a non-click is a real negative. This isolates how well each model
  *orders* a slate. It is an upper bound on reranking quality.
- **End to end** (`2_CG_RR_E2E`): the generator retrieves 450 articles from the whole catalogue and
  the reranker orders them. Most retrieved articles were never shown, so their relevance is unknown:
  they can never be credited as clicks, and a click the generator missed counts as a miss. These are
  known-click metrics. They are lower bounds, and much smaller than the reranker-only numbers.

The two sets of numbers answer different questions and should not be compared directly.

## Reranker-only notebook

`1_RR_only/PE_RR_logged_slates.ipynb` compares three rerankers on the logged test slates:

- **Popularity**: causal smoothed prior article CTR (`RR_popularity_baseline.ipynb`).
- **Content similarity**: entity cosine to the user's causal click profile
  (`RR_content_similarity_baseline.ipynb`).
- **RR4 attention reranker**: the `+metadata` architecture with the `softmax+lambda_ndcg` loss,
  trained inside the notebook exactly as in stage 3 of `RR_4_branch_interaction.ipynb`. The
  preprocessor is fitted on a 10,000-impression training sample. The model trains on every training
  impression (slates capped at 100, 10% holdout) for up to 20 epochs, with early stopping
  (patience 3) on holdout nDCG@10.

It reports MRR, nDCG, recall, engaged-impression rate and observed CTR at `K_VALUES = (5, 10, 30)`,
a breakdown by the user's prior click count, and a comparison with the metrics recorded by the RR_4
notebook. Outputs:

- `5_03_Pipeline_Evaluation/artifacts/reranker_logged_slates/reranker/`: `model.pt` and
  `preprocessor.joblib`
- `5_03_Pipeline_Evaluation/artifacts/reranker_logged_slates/scores/`: one score file per model
- `results/rr_only_logged_slates_<split>.csv`, `..._by_history.csv` and `.json`

The `rr_only_` prefix keeps these out of the `pipeline_*` files that `results.ipynb` compares.

## End-to-end baseline pipelines

Two notebooks rebuild the same baseline candidate generator in memory and differ only in the
reranker. Neither reads or writes intermediate candidate files.

- **Candidate generation**: 200 rolling prior-hour popularity candidates plus 250 content
  text/entity cosine candidates per request (450 in total), with request history removed.
- **Rerankers**:
  - `PE_popularity_reranker_baseline_pipeline.ipynb` scores each candidate by
    `smoothed_prior_article_ctr`, with ties broken by ascending `news_code`.
  - `PE_content_reranker_baseline_pipeline.ipynb` applies the fixed `content_fusion_text_0.50` rule.
    Set `SELECTED_TRIAL = None` to compare all five content rules on validation instead.

Outputs: `results/pipeline_baseline_content_popularity_combined_<reranker>_<split>_<impressions>.csv`,
the final Top-K parquet and config under `5_03_Pipeline_Evaluation/artifacts/`, and causal-feature
caches under `5_03_Pipeline_Evaluation/artifacts/replayed_features/`.

## End-to-end attention pipeline

`PE_attention_reranker_niche_pipeline.ipynb` trains (or loads) the niche two-tower generator from
`CG_niche_content_two_tower_sequential_combined.ipynb` and the `RR_4` `+metadata` reranker, then
generates candidates in memory and evaluates the reranked Top-K.

The reranker trains on the generator's training candidates joined to `labels_train`. A generated
candidate that was never shown has no label and is filled with `clicked = 0` for the listwise loss,
a known false-negative risk. Evaluation still treats unshown candidates as unknown.
`TRAIN_IMPRESSIONS` and `EVAL_IMPRESSIONS` control runtime.

Checkpoints are saved under `5_03_Pipeline_Evaluation/artifacts/attention_reranker_niche/`
(`candidate_generator/model.pt`, `reranker/model.pt`, `reranker/preprocessor.joblib`), and metrics
under `results/pipeline_attention_reranker_niche_generated_<split>_<impressions>.csv` with a
metadata JSON.

## Metrics

End-to-end notebooks report, at each `k` in `K_VALUES`, every metric in
`5_01_Candidate_Generation/cg_evaluation.py` at `M = k` (pooled and impression-mean recall, hit
rate, long-tail recall, cold-article recall, novelty, catalogue coverage, personalisation), plus
MRR@k and nDCG@k from `5_02_Reranking/rr_evaluation.py`. The reranker-only notebook uses
`rr_evaluation.py` on complete logged slates.
