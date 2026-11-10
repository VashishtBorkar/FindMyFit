# FindMyFit: 10-day deep learning project

Scope update: the owner selected two development tracks. See
[DEVELOPMENT_TRACKS.md](DEVELOPMENT_TRACKS.md) for the active deadline strategy.
The contemporary-data experiment below is the rebuild track's first proposed
milestone, not the only presentation option. The fallback track explicitly keeps
the existing dataset, CLIP features, and metric-learning approach.

Proposed scope, prepared 2026-11-10. The owner has approximately 10 days and must
train a model. This is the immediate milestone; PRODUCT_REBUILD_PLAN.md remains
the longer-term direction. No model performance or dataset access is established
by this document.

## Deliverable and research question

Deliver a trained model, reproducible evaluation, and a working image-upload demo.

Research question: **Can a learned complementary-item retriever outperform direct
visual similarity on held-out contemporary retailer outfit relationships?**

Minimum task: upload/select a top and recommend bottoms from a fixed catalog.
Do not claim full outfit compatibility from this experiment. If linked three-piece
looks are accessible early, add top/bottom/shoes completion as a stretch task.

The class prototype need not include live retailer stock, accounts, a wardrobe,
personalization, production hosting, automatic segmentation, or VLM reasoning.
The existing React/FastAPI interface can display results. Local serving and a
recorded backup are sufficient unless the rubric requires public deployment.

## Model to train

Use a frozen fashion image encoder and precompute catalog features. Start with
Marqo-FashionSigLIP; measure that it runs on the available machine before selecting
it. Keep one encoder for the core experiment. SigLIP2 is an optional later ablation,
not a required second implementation.

Train a conditional query network and a separate candidate projection in PyTorch:

```text
top feature + requested category -> query MLP -> query vector
bottom feature                 -> item MLP  -> candidate vector
score = normalized dot product(query vector, candidate vector)
```

This learns a directional complementary relationship instead of forcing every
garment through the same projection and measuring symmetric distance. Exact
dimensions, dropout, and learning rate are development-set choices; keep the heads
small and log parameter count. Training only the heads still trains a neural model;
report the frozen/trainable boundary explicitly.

If multiple context garments become available, a small masked mean-pooling context
network can replace the single-item input, followed by a tiny attention model only
if time allows. This is an extension, not a reason to delay the minimum experiment.

Train to rank an observed styling partner above sampled alternatives of the same
target category. Use a softmax retrieval loss or a pairwise ranking loss. Mask known
additional positives and duplicate variants from the sampled-negative set. Other
unobserved items are noisy alternatives, not established incompatibilities. Do not
propagate an outfit's overall negative label to every pair.

## Data plan and day-one decision

Collect explicit outfit-to-product relationships from a small number of current
sources. Preserve images, source product IDs, color variants, outfit IDs, timestamps,
and original source links. Store raw extracts and a versioned processed manifest.
Manually inspect examples before scaling extraction. Product collection membership
and generic recommendation widgets are not outfit labels.

On day one, audit a few sources and extract a small sample to establish:

- Whether exact top/bottom links can be recovered reliably.
- Whether individual garment images are downloadable and usable.
- How many distinct garment families and independent looks are accessible.
- Source collection dates, provenance, and intended-use permissions.

Aim for hundreds to low thousands of clean examples if accessible, rather than
promising a large new benchmark. Repeated looks and variants do not count as
independent data. This scale may still be too small to generalize; learning curves
and held-out evaluation should reveal that limitation.

Put a hard one-day limit on source exploration. If extraction is unreliable,
reduce to one category relationship and manually verify a compact dataset. If that
still cannot support meaningful training, explicitly revisit the data constraint
with the owner/instructor. Do not silently revert to the old catalog or assume a
newly published paper's dataset contains recently collected garments.

Separate sources for training relationships and an additional demo catalog are
possible, but report catalog shift. Do not present a modern demo catalog as proof
that the training supervision is contemporary.

## Evaluation and credible claims

Required comparisons using the same encoder, images, split, and candidate pools:

1. Random ranking within the target category.
2. Direct cosine similarity in frozen encoder space.
3. The trained complementary retriever.

Compare on the full eligible held-out candidate pool if practical; report its size.
If using sampled pools, fix them across models and report that protocol. Use
Recall@5/10 and MRR for the observed partner, plus training/validation curves.
Known alternative observed partners should be treated consistently across models.

Create item-family-disjoint train/validation/test partitions before generating
examples. Keep duplicate images and variants together. Partition connected outfit
groups where feasible; if a giant component prevents meaningful splitting, report
coverage loss or choose a transparent alternative protocol. Hyperparameters use
validation only. Do not tune on presentation examples or test results.

Small datasets can make strict splits unstable. Report actual independent item
and outfit counts and avoid hiding a weak split behind a headline metric. Add a
retailer or temporal holdout only if enough data exists.

Exact observed-partner retrieval is a proxy: other recommended bottoms may also
work. Add a small blind preference exercise on approximately 20–30 held-out anchors
if raters are available; randomize system placement and report rater/sample counts
and disagreement. An automated VLM judge is optional supplementary evidence.

Evaluate a separate small phone-photo set with manual crop assistance. Disclose
the supported input conditions; this does not establish arbitrary-photo robustness.
Never use training-set uploads as evidence of generalization.

## Ten-day schedule

| Day | Work | Required checkpoint |
|---|---|---|
| 1 | Source audit, environment/encoder check, small linked sample | Data path and top-to-bottom task fixed |
| 2 | Extraction, cleanup, family deduplication, split manifest | Training and held-out examples load correctly |
| 3 | Cache features; implement random/cosine baselines and evaluation | End-to-end baseline metrics saved |
| 4 | Implement neural heads and loss; overfit a tiny training subset | Gradients, target IDs, and scoring verified |
| 5 | Train on the full training split; limited validation tuning | Reproducible checkpoint and learning curves |
| 6 | Compare models, inspect failures, verify leakage checks | Core experiment complete; extend only if stable |
| 7 | Integrate selected model with the existing API/UI | Upload or choose item -> ranked real results |
| 8 | Phone-photo checks and small human evaluation; stretch only if core passes | Results and limitations recorded |
| 9 | Freeze experiments, create figures, report, and slides | Presentation narrative and demo backup ready |
| 10 | Rehearse and fix critical demo failures | Reproducible local demo and recording |

If data runs late, cut stretch features and catalog breadth before cutting
evaluation or presentation preparation. If the model fails to beat cosine, report
that result and investigate data volume, noisy alternatives, leakage, and feature
quality. Negative results do not justify cherry-picking examples.

## Stretch priorities

In order: multi-item context; a small context-aware versus single-anchor ablation;
an encoder comparison; optional VLM-generated query baseline. Avoid attempting
all of these. Automatic segmentation, Text2Outfit reproduction, GAN training,
large-scale crawling, and full platform functionality are outside the core scope.

## Research takeaways checked for this scope

- [Text2Outfit, ICCV 2025](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Zhai_Text2Outfit_Controllable_Outfit_Generation_with_Multimodal_Language_Models_ICCV_2025_paper.pdf)
  learns cross-modal mappings and predicts outfit compositions/embeddings. It is
  architectural inspiration, not equivalent to a prompt-only VLM baseline.
- [MEDAL, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Sanny_MEDAL_multi-modal_MEta-space_Distillation_and_ALignment_for_Visual_Compatibility_Learning_WACV_2026_paper.html)
  motivates category-pair-specific relationships; its full distillation and web-data
  recipe is beyond this milestone.
- [GeCo, Information Sciences](https://www.sciencedirect.com/science/article/pii/S002002552600561X)
  introduces FashionTaobaoTB by adapting an earlier Taobao outfit collection.
  Publication in 2026 is not evidence of contemporary collection. Its top/bottom
  task is relevant, but dataset access/terms and collection dates require checking.
- [OutfitTransformer](https://arxiv.org/abs/2204.04812) motivates contextual completion
  if the minimum task succeeds; reproducing the full model is not required.

## What survives into the product

Reusable outputs: a clean contemporary data manifest, source adapters, frozen
feature cache, trained contextual-retrieval boundary, evaluation harness, and API
integration. The class result establishes an initial experiment; it does not
establish fashion quality across demographics, complete outfits, live shopping,
or real-world photo conditions.
