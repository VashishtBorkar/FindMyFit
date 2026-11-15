# FashionSigLIP compatibility training comparison

Trained and evaluated 2026-11-15. FashionSigLIP improves sampled observed-partner retrieval for all three model families. The shared projector with ranking loss is preferred by mean validation MRR for both encoders.

## Sampled test results

Means ± sample standard deviations across training seeds 42, 43, and 44. Each query has one recorded partner and 31 fixed same-category alternatives.

| Model | Encoder | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|
| Shared projector + contrastive | CLIP | 38.99% ± 0.56 pp | 59.57% ± 1.15 pp | 0.2573 ± 0.0066 |
| Shared projector + contrastive | FashionSigLIP | 43.62% ± 1.04 pp | 65.93% ± 0.59 pp | 0.2834 ± 0.0063 |
| Shared projector + ranking | CLIP | 39.43% ± 0.64 pp | 59.40% ± 0.18 pp | 0.2663 ± 0.0045 |
| Shared projector + ranking | FashionSigLIP | 46.29% ± 0.59 pp | 66.75% ± 1.40 pp | 0.3190 ± 0.0083 |
| Category-aware two towers + ranking | CLIP | 34.39% ± 0.76 pp | 55.06% ± 1.42 pp | 0.2279 ± 0.0067 |
| Category-aware two towers + ranking | FashionSigLIP | 40.05% ± 0.62 pp | 62.91% ± 1.10 pp | 0.2696 ± 0.0011 |

Raw cosine retrieval without training achieved Recall@5 / MRR of **29.38% / 0.2089** for CLIP and **36.50% / 0.2538** for FashionSigLIP. Raw baselines are deterministic, with no training-seed variability.

The preferred FashionSigLIP ranking model gains **6.86 percentage points Recall@5** and **0.0527 MRR** over the matching CLIP model. This is approximately 19.8% higher MRR.

## Full held-out category pools

These pools contain 547 trousers and 1,418 shoes from the retained test partition. Evaluation groups by anchor/category and credits every recorded partner: 1,106 groups versus 1,137 single-partner sampled queries. These are full held-out category pools, not the complete production catalog.

| Model | CLIP MRR | FashionSigLIP MRR | CLIP Recall@5 | FashionSigLIP Recall@5 |
|---|---:|---:|---:|---:|
| Shared projector + contrastive | 0.0274 | 0.0312 | 2.80% | 3.56% |
| Shared projector + ranking | 0.0314 | 0.0469 | 3.61% | 5.61% |
| Category-aware two towers + ranking | 0.0213 | 0.0304 | 2.01% | 3.24% |

Raw full-pool MRR is 0.0188 for CLIP and 0.0325 for FashionSigLIP. Although all FashionSigLIP trained models improve over their matching CLIP counterparts, FashionSigLIP contrastive and two-tower full-pool MRR do not exceed raw FashionSigLIP cosine. The shared ranking model does, reaching **0.0469 ± 0.0015 MRR**. Its multi-positive Recall@5 is **5.61%**, versus **3.61%** for CLIP ranking. Absolute full-pool retrieval remains weak.

## Per-category sampled Recall@5 for shared ranking

| Encoder | Trousers | Shoes |
|---|---:|---:|
| CLIP | 36.67% | 41.11% |
| FashionSigLIP | 44.42% | 47.43% |

## Protocol and model selection

- Nine new FashionSigLIP checkpoints: three model families, each trained from scratch with seeds 42/43/44. Frozen encoders throughout. Matching existing CLIP checkpoints were replayed for comparison.
- Same 126,928 item IDs for both encoders. All vectors finite, nonzero, float32, approximately unit norm, with image hashes present.
- Official Polyvore partitions and exact-image separation. 10,730 training, 578 validation, and 1,137 sampled test queries. Top-to-trousers/shoes only.
- Every new run verified generated candidates against the original saved CLIP manifest before training. Reporting also checked every saved manifest, source-vector digest, category mapping, and validation replay; historical CLIP test scores reproduced within 1e-7.
- Adam, learning rate 0.001, batch 64, maximum 30 epochs, patience five. Best validation-MRR checkpoint reloaded. Contrastive margin 1.0; ranking temperature 0.1. Fixed training alternatives.
- Shared structure changed from 512 → 256 → 128 to 768 → 256 → 128. Outputs remain normalized. No width/depth sweep, fine-tuning, or sampler changes.
- FashionSigLIP shared models have 229,760 parameters versus 164,224 for CLIP; two towers have 468,032 versus 336,960. This is not a parameter-matched encoder comparison.
- Mean validation MRR for shared ranking: CLIP 0.2547; FashionSigLIP 0.3136. Selection was saved before computing new test predictions.

## Checkpoints and serving

The recommended family is FashionSigLIP shared ranking. Seed 42 has the best validation MRR among its three runs (0.3243); its checkpoint is [seed-42/shared_projector.pt](../data/tracks/fallback/runs/fashionsiglip-comparison-001/seed-42/shared_projector.pt). Seeds 43/44 are retained for reproducibility. All new checkpoints record source encoder identity, fingerprint, vector digest, input dimension, and candidate digest.

No database embeddings, application settings, runtime encoder, metric catalog vectors, or FAISS indexes were replaced. Serving this checkpoint requires FashionSigLIP upload encoding with the generation preprocessing, new projected catalog vectors, and new versioned indexes.

## Limits

The historical test set was previously examined during earlier experiments; these results are exploratory rather than a fresh final holdout. Three seeds measure optimization variability, not independent data replications or statistical significance. This experiment does not identify whether gains arise from pretraining, encoder capacity, or the larger projector input. Original FashionSigLIP preprocessing provenance was not independently reconstructed. Sampled alternatives may be compatible, unrecorded positives remain unlabeled, and no human outfit-quality or user-photo evaluation was performed.

## Reproduction

Run the following for training seeds 42, 43, and 44, changing the output directory for each run:

```bash
.venv/bin/python -m training.metric_learning.compare_retrieval \
  --encoder-name marqo-fashion-siglip \
  --encoder-version c56244cc94f92419e8369fa71efdaf403b124ce8 \
  --reference-candidates data/tracks/fallback/runs/controlled-seed-42/candidates.json \
  --seed 42 --training-seed 42 --epochs 30 --patience 5 --defer-test \
  --output data/tracks/fallback/runs/fashionsiglip-comparison-002/seed-42
```

The report expects `seed-42`, `seed-43`, and `seed-44` below its run directory and matching `controlled-seed-*` CLIP runs in its parent:

```bash
.venv/bin/python -m training.metric_learning.report_encoder_comparison \
  --runs data/tracks/fallback/runs/fashionsiglip-comparison-001
```

[Protocol](../data/tracks/fallback/runs/fashionsiglip-comparison-001/protocol.json) · [Selection](../data/tracks/fallback/runs/fashionsiglip-comparison-001/selection.json) · [Summary](../data/tracks/fallback/runs/fashionsiglip-comparison-001/summary.json) · [Per-run CSV](../data/tracks/fallback/runs/fashionsiglip-comparison-001/comparison.csv)

## Verification

Twelve focused unittest checks passed: existing architecture/checkpoint, split/sampling, and resampling/category-head checks, plus new multi-positive full-pool semantics and finite-gradient checks for 768D inputs across all three training families. Real-data startup gradient checks passed for all nine runs. `pytest` is absent from the environment; these unittest-based tests were run using `python -m unittest discover`.
