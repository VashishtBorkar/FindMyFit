# Controlled comparison of compatibility models

All three models were trained from scratch using frozen CLIP ViT-B/32 vectors.
The task is top-to-trousers or top-to-shoes retrieval. Every query has one
recorded outfit partner and 31 fixed same-category alternatives.

## Protocol

- Official Polyvore JSON partitions with exact image hashes isolated across splits.
- 10,730 training queries, 578 validation queries, and 1,137 test queries.
- Candidate sampling seed 42, with initialization and batch-order seeds 42, 43, and 44.
- Adam, learning rate 0.001, batch size 64, maximum 30 epochs, patience five.
- Best checkpoint selected by validation MRR and reloaded before test evaluation.
- Hidden dimension 256 and output dimension 128 for every model.
- Contrastive margin 1.0, with positive and sampled-alternative pairs equally weighted.
- Ranking cross entropy uses temperature 0.1.
- All nine runs have identical candidate and CLIP-vector digests.

## Test results

Values show the mean and sample standard deviation across three training seeds.

| Model | Parameters | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| Shared projector + contrastive loss | 164,224 | 38.99% ± 0.56 pp | 59.57% ± 1.15 pp | 0.2573 ± 0.0066 |
| Shared projector + ranking loss | 164,224 | 39.43% ± 0.64 pp | 59.40% ± 0.18 pp | 0.2663 ± 0.0045 |
| Category-aware two towers + ranking loss | 336,960 | 34.39% ± 0.76 pp | 55.06% ± 1.42 pp | 0.2279 ± 0.0067 |

Frozen CLIP cosine achieved Recall@5 of 29.38% and MRR of 0.2089 on the same candidates.
Expected random Recall@5 is 15.625%.

## Per-category results

| Model | Trousers Recall@5 | Shoes Recall@5 |
|---|---:|---:|
| Shared projector + contrastive loss | 36.67% | 40.41% |
| Shared projector + ranking loss | 36.67% | 41.11% |
| Category-aware two towers + ranking loss | 33.02% | 35.22% |

## Selected checkpoints

| Model | Seed | Best epoch | Epochs run |
|---|---:|---:|---:|
| Shared projector + contrastive loss | 42 | 4 | 9 |
| Shared projector + contrastive loss | 43 | 7 | 12 |
| Shared projector + contrastive loss | 44 | 4 | 9 |
| Shared projector + ranking loss | 42 | 5 | 10 |
| Shared projector + ranking loss | 43 | 4 | 9 |
| Shared projector + ranking loss | 44 | 4 | 9 |
| Category-aware two towers + ranking loss | 42 | 5 | 10 |
| Category-aware two towers + ranking loss | 43 | 9 | 14 |
| Category-aware two towers + ranking loss | 44 | 7 | 12 |

## Interpretation and limitations

The two shared-projector variants have matching architecture and initialization
for each seed, allowing comparison of their training objectives. Two towers
add category conditioning and approximately double the parameter count. Their
results therefore combine architectural and capacity differences.

The contrastive experiment uses observed positive pairs and sampled alternatives.
It preserves the original margin-based learning approach while changing the
label construction and split protocol. Sampled alternatives may also be valid
outfit partners. Missing embeddings reduce coverage, exact-hash checks leave
possible near-duplicates, and the task covers two category relationships.

These scores describe sampled observed-partner retrieval. Human styling quality,
full-catalog retrieval, and photo robustness require further evaluation. Three
seeds measure initialization and batch-order variability on one fixed split.
Hyperparameters were fixed for this experiment and may favor one objective.

The old saved findmyfit/v1 vectors achieved 99.74% Recall@5 on these queries.
Their earlier training exposure is unresolved, so that score has uncertain
held-out status. The controlled comparison uses fresh checkpoints.

## Artifacts and reproduction

Each seed directory contains all three checkpoints, per-epoch histories, metrics,
the audit, candidate sets, and source snapshots.

```bash
.venv/bin/python -m training.metric_learning.compare_retrieval \
  --seed 42 --training-seed 42 --epochs 30 --patience 5 \
  --output data/tracks/fallback/runs/new-comparison-seed-42
```

Repeat with training seeds 43 and 44 while keeping the candidate seed at 42.

[Per-run CSV](../data/tracks/fallback/runs/controlled-comparison/runs.csv)

![Comparison](../data/tracks/fallback/runs/controlled-comparison/comparison.png)

![Validation curves](../data/tracks/fallback/runs/controlled-comparison/validation_curves.png)

[Contrastive architecture experiments](ARCHITECTURE_EXPERIMENTS.md)

[Additional width, depth, and residual experiments](STRUCTURE_EXPERIMENTS.md)
