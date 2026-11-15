# Additional contrastive structure experiments

Nine additional models were trained using frozen 512D CLIP embeddings and 128D normalized outputs.
The plain and residual six-layer networks have identical parameter counts and initial linear weights.
Each residual block computes ReLU(x + Linear(ReLU(Linear(x)))).

Preferred architecture by mean validation MRR across seeds: **Wide 512**.
Selection was saved before scoring the new test predictions. Earlier baseline results had already been examined.

| Structure | Dimensions | Parameters | Validation MRR | Test Recall@5 | Test MRR |
|---|---|---:|---:|---:|---:|
| Baseline | 512 → 256 → 128 | 164,224 | 0.2512 | 38.99% ± 0.56 pp | 0.2573 ± 0.0066 |
| Wide 512 | 512 → 512 → 128 | 328,320 | 0.2541 | 38.96% ± 0.49 pp | 0.2617 ± 0.0049 |
| Wide 1024 | 512 → 1024 → 128 | 656,512 | 0.2485 | 38.70% ± 0.47 pp | 0.2572 ± 0.0063 |
| Plain six-layer | 512 → 256 × 5 → 128 | 427,392 | 0.1842 | 27.59% ± 2.63 pp | 0.1818 ± 0.0153 |
| Residual six-layer | 512 → 256 → 2 residual blocks → 128 | 427,392 | 0.2505 | 38.58% ± 0.42 pp | 0.2552 ± 0.0074 |

Values are means and sample standard deviations over training seeds 42, 43, and 44.

## Interpretation

Increasing the single hidden layer from 256 to 1,024 units quadrupled parameter count without improving mean test Recall@5 or MRR. The 512-unit model remains preferred by validation MRR, although its test Recall@5 is effectively unchanged from the baseline.

The plain six-layer network performs substantially worse under these training settings. Adding identity skip connections at the same parameter count restores most of the baseline performance. This suggests that connectivity and optimization matter here. It does not establish the cause of the plain model's weaker result or prove that deeper networks cannot work.

The practical choice from this sweep is the 512-unit projector when prioritizing validation MRR, or the 256-unit baseline when prioritizing model size. The observed differences among baseline, wide, and residual models are small relative to the variation across seeds.

## Protocol

All runs use identical saved candidates, official Polyvore splits, exact-image separation, and frozen CLIP vectors.
There are 10,730 training, 578 validation, and 1,137 test queries for tops paired with trousers or shoes.
Each query has one recorded partner and 31 sampled same-category alternatives.
Adam uses learning rate 0.001, batch size 64, contrastive margin 1.0, a 30-epoch limit, and patience five.
The checkpoint with the best validation MRR is reloaded. Test scoring is deferred until architecture selection.
The baseline and 512-wide results reuse previous runs. Nine new checkpoints cover the three new structures.

## Limits

Training seeds measure optimization variability on one fixed split. They are not independent data replications.
This is recorded-partner retrieval from 32 candidates. Other candidates may also be compatible.
Full-catalog retrieval and styling quality on user photographs remain unmeasured.
The same learning rate and margin were used for every structure. Architecture-specific tuning could change results.

## Reproduce

```bash
.venv/bin/python -m training.metric_learning.compare_retrieval \
  --models contrastive_wider contrastive_plain_six contrastive_residual \
  --seed 42 --training-seed 42 --epochs 30 --patience 5 --defer-test \
  --output data/tracks/fallback/runs/new-structure-seed-42
```

Repeat with training seeds 43 and 44. Candidate seed remains 42.

[Detailed metrics](../data/tracks/fallback/runs/structure-comparison/summary.json)

![Structure comparison](../data/tracks/fallback/runs/structure-comparison/comparison.png)

![Learning curves](../data/tracks/fallback/runs/structure-comparison/learning_curves.png)
