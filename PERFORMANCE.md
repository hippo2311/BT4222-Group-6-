# Feature replay experiment

Branch: `perf/feature-replay`. The audited v2 full build remains at `4_Feature_Engineering/artifacts/prequential-all/`. This branch is a prototype; the v2 audit does not certify its new outputs.

## Changes

- Draw the same training negatives before expensive feature calculation. Calculate training features for selected pairs, while updating temporal state from **every** allowed logged event. Validation/test slates remain complete.
- Apply SQLite updates after every timestamp, but commit every 1,000 updates and at close. The same connection sees pending writes; this does not delay feedback. Scratch state is not resumable after interruption.
- Keep full-slate calculation when slate context or selected custom notebook features are enabled. Unknown feature callbacks also use the conservative full-slate path.
- Preserve coverage over all logged exposures. Feature-quality ranges/zero fractions now describe **written** rows: sampled training and complete held-out slates. The reports identify this scope explicitly.
- Record replay-phase timings in `replay_timing.csv`. Keep feature formulas, history pooling, same-second handling and negative-sampling rules unchanged.

No incremental user sums or half-life/profile loop vectorisation is included. A 1,000-timestamp profile pointed mainly to DataFrame construction/indexing/joins; early-window half-life estimates were rarely supported. Later dense periods may have different bottlenecks.

## Small-run evidence

A 3,000-timestamp chronological prefix, ending 9 November 2019 at 01:42:31, processed 153,408 logged pairs and wrote 27,027 training rows. Separate fresh processes measured replay plus buffered writer finalisation:

| Original driver | Prototype | Observed gain |
|---:|---:|---:|
| 41.63 s | 38.02 s | 1.09×; about 9% less time |

This is one sparse-prefix comparison, not a prediction for the full run. The initial same-process trial showed about 5%; timing noise/cache effects matter. No full-scale memory claim is made.

Selected pair keys and labels matched exactly. Feature values and request profiles matched both the original replay and the saved full v2 prefix within `rtol=1e-6, atol=1e-7`. All negative groups and full-log coverage matched. The full manifest SHA-256 stayed unchanged. The prefix contains training only; automated fixtures separately exercise the shared train/validation second, frozen feedback, full-slate fallback and dropped-negative history/eligibility.

Local evidence: `results/optimization/compare-3000/comparison.json`, phase timings in the optimized run, and `results/optimization/baseline-profile/profile.txt`. Results are gitignored.

## Run it

From the project root, after activating `.venv`, reproduce the small comparison:

```bash
PYTHONPATH=src python scripts/benchmark_feature_replay.py --timestamps 3000
# Optional profiling of both variants (profiler overhead affects timings):
PYTHONPATH=src python scripts/benchmark_feature_replay.py --timestamps 1000 --profile
python -m pytest -q
```

The benchmark writes only under `results/optimization/` and links existing processed data and representations. `--reuse` compares existing benchmark outputs without rerunning. The reference driver is pinned to commit `e9b1ba1`; current notebook formulas are used, then checked against the saved v2 prefix when available.

For the full experiment, run **stage 4 only**, using existing representations:

```bash
python run_all.py --from-stage 4 --to-stage 4 --force
```

On this branch, full output goes to `4_Feature_Engineering/artifacts/prequential-all-optimized/`; smoke output adds `-smoke-N`. The runner refreshes source-notebook outputs. The original full artifact directory is preserved. Restart an existing notebook kernel before using edited helpers.

After the full experiment, compare all selected pairs/labels, features and profiles against v2, then redo readiness/leakage checks before adopting the prototype for model work. Any feature differences need investigation and a new version/audit. Model baselines and column-selection ablations can continue using the audited v2 data meanwhile.
