# Training audit and next experiments

Reviewed 2026-11-15. This audit inspected the current implementation, saved run metadata and histories, real Polyvore records, all stored CLIP vectors, and selected trained checkpoints. It did not train new models. Diagnostics are saved in `data/tracks/fallback/runs/training-audit-2026-11-15.json`.

## Main findings

The corrected experiments form a defensible baseline for observed-partner retrieval. More epochs or a larger projector are not supported as the next default intervention. Improve supervision variety, establish stronger evaluation, and tune the objective before expanding architecture.

The current task is deliberately narrow: a top as the query, with trousers or shoes as the requested category. It does not establish full-outfit compatibility or robustness to uploaded photos.

## Which training pipeline was used?

Recent experiments used `python -m training.metric_learning.compare_retrieval`. They preserved official partitions, removed exact-image overlap, excluded known partners from sampled alternatives, and reloaded the best validation-MRR checkpoint.

The README still directs users to `python -m training.metric_learning.train`. That legacy entry point still expands outfit labels into all ordered pairs, randomly splits pairs, reports accuracy at distance 0.5, and tests final rather than reloaded best weights. A negative outfit does not imply that every constituent pair is incompatible. Reversed pairs and related items can cross its random pair splits. Those problems remain in that entry point, but do not describe the recent controlled runs. Deprecate or clearly mark the legacy command before further work.

## Audit by stage

| Stage | What happens now | Assessment and next action |
|---|---|---|
| Source data | Official historical Polyvore outfit JSON files | Appropriate for a class benchmark. Recency and product-photo bias constrain claims about real users. |
| Coverage | Training references 114,806 item records; 63,425 have vectors | Only 55.2% coverage. Investigate missing files, ID mapping, and filtering before assuming all require re-encoding. Expand coverage as a separate experiment. |
| Positive labels | Cross-category items appearing together become positive pairs | Co-occurrence is weak compatibility evidence, conditional on the omitted outfit. Not human preference or proof that all pairs work independently. |
| Splits | Official partitions; exact hashes reserved for test, then validation, then train | No retained exact-image overlap. Audit also found zero shared parsed Polyvore product IDs among available records across splits. Near-duplicates and shared styles/creators remain unmeasured. |
| Features | Frozen 512D CLIP vectors from SQLite | All 126,928 vectors finite, unit norm to floating-point precision, with hashes present. No obvious norm mismatch with current upload encoding. Original image-to-vector provenance and photo-domain robustness remain unverified. |
| Scope | Top-to-trousers and top-to-shoes only | 10,730 queries from 6,217 distinct anchor items. This is intentional scope reduction, not all usable Polyvore supervision. |
| Alternatives | 31 fixed, random, same-category alternatives per query | Reproducible and category-controlled. Broad overall coverage, but each query repeats the same alternatives every epoch. Not confirmed negative labels. |
| Multiple positives | Each partner becomes a separate query; other known partners excluded from alternatives | No observed-partner-as-negative contradiction in prepared candidates. However, 1,409 of 8,693 anchor/category groups have multiple partners; query weighting favors groups with more positives. Consider grouped positives or balanced sampling. |
| Architecture | Shared normalized projector, or independent query/item towers | Shared geometry imposes one representation for all relationships. Two towers change weight sharing, category conditioning, and capacity together, so their worse result cannot isolate the cause. |
| Loss | Balanced margin contrastive, or temperature-0.1 cross entropy | Contrastive formula independently matched to balanced pairwise loss. Both use noisy sampled alternatives. Tune margin or temperature before interpreting objective differences as definitive. |
| Optimization | Adam, LR 0.001, batch 64, no weight decay or LR schedule | Reasonable baseline, under-tuned across architectures. Equal epochs are equal optimizer-step budgets at fixed batch size, but unequal compute across model sizes. |
| Stopping | Up to 30 epochs, patience five, best validation MRR reloaded | Correct for the selected objective. Many models peak early while training loss keeps improving. Lower LR or scheduling is a hypothesis; blindly adding epochs is not. |
| Evaluation | Recorded-partner rank among one positive and 31 sampled alternatives | Correct sampled retrieval metric, not binary compatibility accuracy or full-catalog retrieval. Add full held-out category pools and multi-positive metrics. |
| Model selection | Three initialization/batch-order seeds; one candidate seed and one data partition | Measures optimization variability only. Repeated inspection of test scores makes future iterative gains exploratory. Use validation for all subsequent decisions. |
| App integration | Runtime expects a shared projector and catalog vectors from that same checkpoint | New checkpoints have not automatically replaced old vectors/indexes. Two towers need a dedicated inference adapter. Re-embed/version/reindex before demo claims. |

## New measured diagnostics

Training examples contain 10,730 distinct positive pairs and 332,134 distinct anchor/alternative pairs (332,630 occurrences). The concern is repeated per-query exposure and noisy labels, not a dataset consisting of only 31 negatives. Increasing epoch count alone does not create new supervision.

Training has 4,445 top-to-trousers and 6,285 top-to-shoes queries. Validation has 220/358, and test has 430/707. Report both per-category and equal-category averages alongside the overall query-weighted score.

The table below uses seed-42 selected checkpoints in evaluation mode on their saved training candidates. Validation values are taken from that checkpoint's recorded best epoch.

| Model | Training Recall@5 | Validation Recall@5 | Median positive distance | Median sampled-alternative distance |
|---|---:|---:|---:|---:|
| Shared contrastive | 43.27% | 36.51% | 0.433 | 0.553 |
| Shared ranking | 48.56% | 34.78% | 0.686 | 0.813 |
| Two towers, ranking | 40.52% | 31.83% | 1.302 | 1.358 |

These gaps suggest generalization limitations but do not isolate their cause, because training and validation differ in garment distribution and candidate pools. The low training performance of the plain six-layer seed-42 selected checkpoint (26.26%) also cautions against describing every weak model as simply overfitting.

For the shared contrastive checkpoint, **99.978% of sampled alternatives are inside margin 1.0** and contribute to the negative hinge term. Therefore, a proposed explanation that most random alternatives have become inactive is not supported for this checkpoint. Positive and alternative distance distributions overlap substantially. Distance comparisons across different objectives/tower spaces are descriptive, not calibrated compatibility scores.

In seed 42, shared-ranking validation MRR peaked at 0.2553 at epoch 5 and fell to 0.2279 at epoch 10, even as epoch training loss fell from 3.0461 to 2.8088. More fitting of the current comparisons did not improve validation retrieval.

## Experimental controls that need improvement

1. Save validation/test candidate manifests once and reuse them independently of training sampling. Current preparation shares one RNG across partitions. Changing candidate count or training rows can change downstream validation/test draws. Comparing runs with different `--candidates` directly changes both training and evaluation difficulty.
2. Use separate seeds for initialization, batch order, and training alternative resampling. Paired comparisons should share initialization and evaluation conditions.
3. For 32 versus 128 training candidates, fix the positive queries, batch size, optimizer steps, and evaluation manifests. Report wall-clock cost too. In balanced contrastive loss the total alternative weight stays constant; in cross entropy the denominator and objective difficulty change.
4. Record validation loss, evaluated training retrieval, positive/alternative score distributions, active hinge fraction, gradient norms, and learning rate. Existing histories only save epoch training loss and validation retrieval. Current batch checks establish finite gradients at startup, not their quality throughout training.
5. Keep retrieval reports separate from user-photo testing and human judgments. Single-positive Recall@5 here is effectively a hit rate. For full pools, credit all known partners and report MRR plus multi-positive Recall/nDCG. Unknown compatible alternatives remain unlabeled even then.
6. Repeated queries can share anchors. Confidence intervals should group by anchor (or source outfit), rather than treating every query as independent. Three seed standard deviations are not significance tests.
7. Retain the old test set for historical comparison and call further tuning exploratory. A fresh final holdout must be independent of model selection, preferably with deduplicated images and reviewed compatibility labels. Resampling candidates on already examined test items does not create a fresh item holdout.

## Prioritized experiments and falsifiable hypotheses

| Priority | Change | Hypothesis | Comparison and success criterion |
|---|---|---|---|
| 0 | Freeze evaluation manifests; add full-category evaluation and diagnostics | Some apparent gains may depend on the small sampled pool or candidate draw | Re-score existing checkpoints first. Establish validation MRR and per-category full-pool ranks before changing training. This improves measurement, not necessarily model quality. |
| 1 | Resample 31 training alternatives each epoch | Broader per-query exposure reduces reliance on specific alternatives and improves held-out retrieval | Fixed versus resampled 32-candidate training, same model/loss/init/steps. Seek higher validation MRR and full-pool performance across three seeds. |
| 2 | Increase to 64 or 128 training candidates with resampling | More simultaneous comparisons improve ranking beyond variety alone | Resampled 32 versus 64/128 with identical evaluation. Report quality and compute. Improvement is not guaranteed because unknown compatible garments can be penalized. |
| 3 | Small learning-rate and objective-parameter search | Current settings are suboptimal for individual objectives, particularly the independent towers | Start LR 0.0001/0.0003/0.001; separately vary ranking temperature 0.05/0.1/0.2 or contrastive margin 0.5/1.0/1.5. Select on validation, replicate finalists. Avoid a large Cartesian search. |
| 4 | Group positives, or balance sampling per anchor/category | Repeated anchors and sparse positive labels distort training weights | Compare present query weighting to equal group weighting; optionally multi-positive ranking. Evaluate category/anchor macro metrics as well as overall MRR. |
| 5 | Shared backbone with small category heads | Relationship-specific geometry improves matching while sharing statistical strength | Compare category-conditioned shared architecture to unconditioned shared and independent towers at similar parameter budgets. Keep loss and sampling fixed. |
| 6 | Triplet/relative ranking versus absolute-distance contrastive | Relative ordering fits complementary matching better than pulling all partners close | Same architecture and alternative sets; tune triplet margin on validation. Improvement must appear in held-out ranks, not only lower loss. |
| 7 | Random plus moderately difficult alternatives | Harder comparisons improve fine discrimination after random resampling gains saturate | Sample only from training items, exclude known partners, retain random component, and inspect mined examples. Test against random resampling with equal candidate count. Active-margin diagnostics show inactivity alone is not the current justification. |
| Later | Recover missing items, add category relationships, or replace CLIP features | More positive relationships or better visual information improve generalization | Separate these data/feature experiments from sampler and architecture changes. New encoders need original images and feature recomputation. Wider category coverage needs per-relationship evaluation. |

## Practical next round

Use the shallow shared-ranking model and the original shallow contrastive model as references. First compare fixed versus epoch-resampled alternatives with 32 total candidates. Then increase training candidates for the better-supported objective, followed by a small learning-rate/loss-parameter search. Hold evaluation fixed throughout. Avoid launching another width/depth sweep until these controls are in place.

Run a longer, lower-learning-rate training experiment only as a controlled optimization test with validation tracking. Do not automatically disable early stopping or train on held-out items. More epochs, more sampled alternatives, more positive pairs, and more image information are different interventions.

## Verification and provenance

All seven focused architecture/checkpoint and split/sampling tests passed. An independent tensor calculation verified candidate contrastive loss equals the intended equally weighted positive/alternative pairwise loss. Saved checkpoint diagnostics use seed 42 only and are not claims about all seeds. Product identity overlap checks parse `tid` from the source item image URL and cover records with available embeddings; exact URL identity does not detect every visual duplicate.

No training, checkpoint, database vector, index, or application setting was changed by this audit. The legacy command and candidate RNG coupling are surfaced for correction rather than silently altered during review.
