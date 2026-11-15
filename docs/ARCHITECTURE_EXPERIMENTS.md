# Contrastive architecture experiments

The shared projection network was widened, deepened, and regularized while
keeping frozen CLIP features, 128-dimensional outputs, the contrastive objective,
and the existing audited retrieval protocol. Nine new checkpoints were trained.

Widening increased mean test MRR from 0.2573 to 0.2617 while Recall@5 stayed
close to 39%. Adding another hidden layer produced similar results to the
baseline. The combined LayerNorm/dropout variant scored lower on both metrics.
The differences between the baseline and wider/deeper variants are small relative
to variation across seeds.

## Validation selection

The preferred structure by mean validation MRR was **Wider**.
The selection was written to selection.json before scoring the new models on test.

| Architecture | Layer dimensions | Parameters | Validation MRR |
|---|---|---:|---:|
| Retrained baseline | 512 → 256 → 128 | 164,224 | 0.2512 ± 0.0004 |
| Wider | 512 → 512 → 128 | 328,320 | 0.2541 ± 0.0043 |
| Deeper | 512 → 512 → 256 → 128 | 426,880 | 0.2493 ± 0.0059 |
| Deeper + LayerNorm/dropout | 512 → 512 → 256 → 128 | 428,416 | 0.2424 ± 0.0026 |

## Test results

Values show means and sample standard deviations across training seeds 42, 43, and 44.

| Architecture | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Retrained baseline | 38.99% ± 0.56 pp | 59.57% ± 1.15 pp | 0.2573 ± 0.0066 |
| Wider | 38.96% ± 0.49 pp | 59.57% ± 0.85 pp | 0.2617 ± 0.0049 |
| Deeper | 38.84% ± 0.48 pp | 59.22% ± 1.54 pp | 0.2577 ± 0.0019 |
| Deeper + LayerNorm/dropout | 36.65% ± 0.40 pp | 56.52% ± 0.88 pp | 0.2422 ± 0.0039 |

## Per-category Recall@5

| Architecture | Trousers | Shoes |
|---|---:|---:|
| Retrained baseline | 36.67% | 40.41% |
| Wider | 35.89% | 40.83% |
| Deeper | 34.73% | 41.35% |
| Deeper + LayerNorm/dropout | 35.43% | 37.39% |

## Training

- Official Polyvore partitions, exact image hashes isolated across splits, and identical saved candidate sets.
- 10,730 training, 578 validation, and 1,137 test queries, covering top-to-trousers and top-to-shoes.
- Thirty-epoch limit, patience five, and checkpoint selection by validation MRR.
- Adam at learning rate 0.001, batch size 64, and contrastive margin 1.0.
- Positive pairs and sampled alternatives have equal total loss weight.
- The regularized variant adds LayerNorm and dropout 0.15 after each hidden linear layer.
- Existing default-model behavior and legacy checkpoint loading passed regression checks.

| Architecture | Seed | Selected epoch | Epochs run |
|---|---:|---:|---:|
| Retrained baseline | 42 | 4 | 9 |
| Retrained baseline | 43 | 7 | 12 |
| Retrained baseline | 44 | 4 | 9 |
| Wider | 42 | 4 | 9 |
| Wider | 43 | 4 | 9 |
| Wider | 44 | 5 | 10 |
| Deeper | 42 | 4 | 9 |
| Deeper | 43 | 4 | 9 |
| Deeper | 44 | 7 | 12 |
| Deeper + LayerNorm/dropout | 42 | 6 | 11 |
| Deeper + LayerNorm/dropout | 43 | 7 | 12 |
| Deeper + LayerNorm/dropout | 44 | 9 | 14 |

## Interpretation limits

The changes alter network capacity, depth, and regularization. LayerNorm and
dropout were introduced together in one variant, so their individual effects
are unresolved. Hyperparameters were held fixed and may suit architectures
differently. Results cover one data split and three training seeds.

Evaluation measures recorded-partner retrieval from 32 candidates. Alternative
items can also be compatible, and human styling quality remains unmeasured.
Performance on arbitrary user photos and the complete catalog remains unmeasured.
Exact-hash filtering leaves possible near-duplicate images.

## Artifacts

[Selection and scores](../data/tracks/fallback/runs/architecture-comparison/summary.json)

[Per-run CSV](../data/tracks/fallback/runs/architecture-comparison/runs.csv)

![Architecture comparison](../data/tracks/fallback/runs/architecture-comparison/comparison.png)

The training reports retain their deferred test fields. Final test scores are
saved in the architecture-comparison summary after validation selection.

## Reproduction

```bash
.venv/bin/python -m training.metric_learning.compare_retrieval \
  --models contrastive_wide contrastive_deep contrastive_regularized \
  --seed 42 --training-seed 42 --epochs 30 --patience 5 --defer-test \
  --output data/tracks/fallback/runs/new-architecture-seed-42
```

Repeat for training seeds 43 and 44 while retaining candidate seed 42. Each
checkpoint stores its architecture settings for runtime loading.

![Validation curves](../data/tracks/fallback/runs/architecture-comparison/validation_curves.png)

[Additional width, depth, and residual experiments](STRUCTURE_EXPERIMENTS.md)
