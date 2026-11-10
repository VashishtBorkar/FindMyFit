# FindMyFit development tracks

Owner decision: pursue the broader rebuild on one branch and improve the existing
implementation on another. Approximately 10 days remain before the deep learning
class presentation, which requires training a model. The existing dataset is
explicitly authorized for the fallback. The rebuild continues after the class.

## Branches

| Branch | Purpose | Data/model direction |
|---|---|---|
| `main` | Preserve the current baseline | Existing committed implementation |
| `improve/class-project-fallback` | Produce a dependable class submission | Existing dataset, CLIP features, and metric-learning approach |
| `refactor/findmyfit-v2` | Develop the long-term recommendation/product rebuild | Contemporary outfit data, new compatibility modeling, later platform features |

Both development branches begin with the same planning documents and existing
application. No model improvements have been implemented merely by creating them.
No remote push or public deployment is part of this setup.

## Work allocation and presentation decision

Secure a working fallback early, then spend the remaining experiment time on the
rebuild. Keeping both branches active does not double the available development
time. Avoid two UI rewrites or two simultaneous infrastructure migrations.

| Window | Fallback track | Rebuild track |
|---|---|---|
| Days 1–2 | Verify artifacts, capture baseline, prepare reproducible training/evaluation | Time-box source audit and choose first experiment |
| Days 3–4 | Retrain and compare one modest improvement; verify application | Collect/clean contemporary sample and cache features as capacity allows |
| Days 5–6 | Freeze a working checkpoint, results, and demo | Train/evaluate the proposed replacement |
| Day 7 | Confirm fallback remains runnable | Decide whether rebuild evidence and demo are ready to present |
| Days 8–10 | Fix critical issues only; prepare slides, report, and backup recording | Freeze presentation version if selected; defer unfinished research |
| After class | Retain reproducible baseline | Resume broader model/data/product work |

These are relative checkpoints, not promises that both experiments fit. If time
slips, protect fallback training/evaluation and presentation preparation first.

Present the rebuild only if it has a reproducible trained checkpoint, an honest
held-out evaluation, and a stable demo. It need not achieve exceptional fashion
quality. Otherwise present the improved existing approach and explain the rebuild
as future work. Keep quantitative claims and datasets clearly separated if showing
both; different catalogs and splits do not support a direct improvement claim.

## Fallback scope

Keep the existing model family, dataset, and application. Prioritize scientific
correctness and reproducibility over architecture changes:

1. Verify access to the catalog, original outfit labels, images, CLIP weights,
   metric checkpoint, and compute. Those large artifacts are absent from Git;
   branch creation does not supply them.
2. Save a baseline run and preserve its checkpoint before retraining.
3. Add explicit seeds and split manifests. At minimum keep reversed copies of a
   pair in the same split and account for outfit grouping. Prefer item-family-
   disjoint evaluation where feasible; document residual overlap and data loss.
4. Review label contradictions and duplicate pairs. Do not claim that inherited
   outfit labels are reliable pair labels; measure and disclose this limitation.
   Any sampling adjustment should be a separately evaluated experiment.
5. Reload the best validation checkpoint before final test evaluation. The present
   training script saves the best checkpoint but tests the final in-memory model.
6. Bound tuning time and isolate Optuna studies by data/split configuration. Avoid
   reusing a previous study's results as if they came from the current split.
7. Try one small model change, such as dropout/weight decay in the current
   projector. Compare with the unchanged projector on the same corrected split;
   do not promise improvement before measuring it.
8. Save learning curves, retrieval examples, relevant metrics, and limitations.
9. Regenerate metric catalog embeddings and indexes for the selected checkpoint,
   then verify the API/UI actually uses it.

Do not compare an old score on a leaky split with a new score on a corrected split
and call the difference a model improvement. A fair comparison may require
retraining both projector variants under the corrected protocol. Lower reported
accuracy after leakage fixes can reflect more credible evaluation.

Defer encoder replacement, model-family changes, automatic segmentation, current
retailer feeds, account/wardrobe features, and production deployment on this track.

## Rebuild scope

Continue PRODUCT_REBUILD_PLAN.md with CLASS_PROJECT_PLAN.md as the first suggested
experiment. Contemporary data access, category-aware complementary retrieval,
and whole-outfit modeling remain research work. Start with one measurable task;
the 10-day checkpoint does not define the final product scope.

Do not introduce a legacy-dataset dependency on this branch just to meet the class
deadline. The fallback branch exists for that purpose. Product functionality may
follow once recommendation experiments are useful.

## Artifact isolation

Git branches isolate tracked files, not ignored files. Switching branches in one
checkout still leaves local databases, model files, indexes, caches, and `.env`
files in place. Use separate worktrees when actively developing both tracks if
convenient, and keep artifacts in distinct directories regardless.

Suggested layout (not created by this document):

```text
data/tracks/fallback/catalog.db
data/tracks/fallback/indexes/
data/tracks/fallback/pairs/
data/tracks/fallback/runs/
checkpoints/tracks/fallback/

data/tracks/v2/catalog.db
data/tracks/v2/indexes/
data/tracks/v2/manifests/
data/tracks/v2/runs/
checkpoints/tracks/v2/
```

Configure database, checkpoint, index, pair-cache, and Optuna paths separately.
The existing immutable image collection may be shared read-only; never mutate a
shared catalog during embedding generation or migration. Clone the legacy SQLite
catalog with a consistent backup procedure before fallback experiments.

Use separate model versions for changed checkpoints, with matching artifact
fingerprints, projected vectors, and indexes. A new checkpoint cannot safely use
an old model's metric vectors. Record each run's code commit, encoder/checkpoint,
data snapshot, split, settings, and metrics. Keep baseline artifacts recoverable.

Shared fixes can be cherry-picked deliberately. Do not automatically merge the
rebuild into the fallback near the deadline. After class, carry useful training
and evaluation fixes forward without importing abandoned legacy assumptions.
