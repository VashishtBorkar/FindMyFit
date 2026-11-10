# FindMyFit product rebuild plan

Status: proposed plan, not an implementation commitment.
Prepared: 2026-11-10. Builds on the current repository and supersedes the earlier
refactor plan as a proposed product direction. The earlier plan remains useful as
historical context.

Immediate priority: a deep learning class presentation in approximately 10 days,
including training a model. See [DEVELOPMENT_TRACKS.md](DEVELOPMENT_TRACKS.md) for
the two-branch strategy and [CLASS_PROJECT_PLAN.md](CLASS_PROJECT_PLAN.md) for a
proposed initial rebuild experiment. The broader rebuild continues during and
after that milestone; an improved existing implementation is the presentation
fallback. Neither class prototype is a production launch.

## 1. Product direction

Build a fashion platform with two distinct, connected features:

1. **Style my wardrobe:** users add clothes they own and build outfits from those
   clothes, with help generating and refining combinations.
2. **Find pieces for this item:** users upload a garment or select one from their
   wardrobe and receive recommendations from a current external catalog.

The second feature is interpreted here as shoppable external recommendations;
confirm this interpretation before committing to catalog partnerships.

The shared value proposition is helping someone decide what to wear or buy around
clothes they actually own. The intended audience includes people who want help
dressing better and fashion enthusiasts seeking inspiration. Demographic reach
can remain broad, while the first evaluation cohort and catalog coverage should
be specific enough to measure quality. Do not infer preferences from the existing
dataset.

Owner constraints: launch soon without a fixed deadline, keep deployment costs
low, prefer open-source models and self-hosted inference, and consider paid model
APIs only if an experiment demonstrates substantial value. A numerical spending
ceiling, supported market, and available weekly development time remain open.
Free hosting tiers are an opportunity to verify, not an assumption that model
inference or image storage will be free.

Both features belong in the product direction. Implement one shared garment
pipeline, then release each workflow incrementally. Users should get value after
adding a few garments; uploading an entire wardrobe must not be an onboarding
requirement. Offer batch entry once single-item entry works reliably.

## 2. Why the present system underperforms

These observations come from code inspection and the owner's report of poor
recommendations. Their relative impact needs measurement.

| Problem | Evidence or mechanism | Replacement direction |
|---|---|---|
| No outfit context | Each candidate is scored against one uploaded image | Condition recommendations on the current outfit and user intent |
| Incorrect pair labels | `load_pairs` copies the outfit label to every ordered pair | Preserve outfit supervision; collect direct pair or preference labels separately |
| Weak negative semantics | An incompatible outfit need not contain only incompatible pairs | Use curated alternatives and graded or preference judgments |
| Restrictive compatibility representation | One projection and Euclidean distance for every category and context | Separate candidate retrieval from contextual outfit ranking |
| Evaluation leakage risk | Both pair orders are generated before random pair splitting | Split by outfit/item groups before constructing examples |
| Outdated recommendations | Owner reports a catalog sourced from a 2017 dataset | Maintain a current product inventory with freshness and availability |
| Input domain mismatch | Whole user photo is passed directly through CLIP | Garment selection, crop/mask, image-quality checks, and real-photo evaluation |
| Product semantics are missing | Current catalog has no product variants, offers, or user ownership | Introduce garments, wardrobe ownership, products, offers, and outfits |
| Misleading match percentages | Displayed percentages are transformations of distance | Use rank and grounded descriptions; measure calibration before confidence claims |

Shared-space distance also encourages transitive closeness: if A is close to B
and B is close to C, A and C cannot be arbitrarily far apart. Styling relationships
need not follow that pattern. A newer encoder alone does not fix the target.

Older clothing is not inherently undesirable. The product should distinguish
catalog freshness, availability, vintage preference, and the user's taste.

## 3. Keep, change, and retire

Keep the repository and replace parts behind stable boundaries. There is no
evidence that rewriting the stack will fix recommendation quality.

| Area | Decision |
|---|---|
| React/Vite | Keep; organize around wardrobe, styling, discovery, and account flows |
| FastAPI/Python | Keep as application API and ML integration layer |
| Settings and typed API schemas | Keep and extend |
| Embedding generation/inference separation | Keep; remove fixed encoder and dimension assumptions |
| FAISS and exact-search parity tests | Keep as baseline and available retrieval implementation |
| SQLite | Preserve legacy catalog and local experiments; plan a versioned migration to a multiuser relational store for launch |
| Image storage boundary | Keep the concept; add private object storage and derived assets |
| Existing tests | Keep useful operational checks; add product and recommendation-quality evaluation |
| Current metric checkpoint | Retain as a benchmark baseline, then remove from the primary product path |
| Old catalog | Retain separately for reproducible experiments; exclude from live shopping inventory |
| Pair-generation assumptions | Replace for future training |
| Current outfit view | Reuse visual assets where useful; replace client-only slot cycling with explicit outfit state |
| Streamlit | Keep as an internal experiment tool if useful |

No bulk deletion or destructive data migration is part of this plan.

## 4. User journeys

### Style my wardrobe

1. Add an item using a flat-lay photo, hanger photo, product photo, or worn photo.
2. Select the intended garment if the image contains several pieces.
3. Confirm the crop/mask and editable category, color, and other useful attributes.
4. Save the garment privately to the wardrobe.
5. Select an anchor item, occasion, and optional style preferences.
6. See a few complete outfits made only from owned, available items.
7. Lock pieces, replace one slot, exclude an item, or save the outfit.

If the wardrobe cannot support a complete outfit, show the useful partial outfit
and what is missing. Offer external discovery explicitly; do not silently insert
products into an owned-clothes result.

### Find pieces for this item

1. Upload a garment or select one from the wardrobe.
2. Choose desired categories, style/occasion, budget, and market; add size filters
   where reliable product data supports them.
3. Receive outfit suggestions with actual catalog products and current offers.
4. Lock a preferred piece and regenerate the remaining suggestions using all
   locked pieces as context.
5. Save the look or open the retailer's product page.

The experience should distinguish a complete look from alternative items for one
slot. A purchased product is not automatically an owned wardrobe item.

### Supporting features for the first usable product

- Accounts, private wardrobe, and editable garment details.
- Saved outfits and collections.
- Lock/swap controls and preference feedback with optional reasons.
- Useful filters and clear states for processing, missing inventory, and failures.
- Deletion of uploaded images, derived assets, and account data.

Social feeds, messaging, virtual try-on, outfit calendars, and elaborate trend
discovery can follow evidence of use. They are not prerequisites for validating
the recommendation promise.

Use two lightweight intent choices rather than separate products: a guided option
with wearable defaults and simple explanations, and an inspiration option with
more variety and editable styling direction. Initially these can change retrieval
constraints and diversity rather than requiring two trained models. Measure
quality separately for both user needs; an aggregate score can conceal failure
for one group.

## 5. Shared garment understanding

Proposed pipeline:

```text
upload -> validate/orient/resize -> select garment -> crop and optional mask
       -> extract editable attributes -> encode -> save derived assets
```

Start with user-assisted cropping as the dependable fallback. Compare that with
automatic detection and segmentation rather than assuming masks always improve
results. Keep the original, crop, and optional mask as separate assets. Some
garments lose useful texture, edges, or silhouette when segmentation fails.

Evaluate candidate segmentation tools such as SAM 3 or a detector plus SAM 2.
These are experiment candidates, not a selected deployment stack [1, 2]. Compare
fashion-focused embedding candidates such as Marqo-FashionSigLIP with the current
CLIP baseline [3]. Retrieval benchmark results do not establish outfit quality.

Build a real input test set covering flat lays, hangers, worn garments, clutter,
lighting changes, occlusion, and multiple garments. Include repeated photos of
the same owned item under different conditions. Measure whether the user selected
the right item and whether its relevant details survived preprocessing.

Segmentation removes some background interference; it does not reconstruct hidden
fabric, undo pose deformation, or solve consumer-to-product domain shift. Evaluate
those errors separately. Show uncertain attributes for correction. Do not infer
fit or exact material from a photo with unwarranted certainty.

## 6. Current inventory and training data are separate systems

The shopping inventory answers **what can we recommend now?** Training and
evaluation data answer **what combinations and preferences should we learn?**
An item list does not provide compatibility labels.

### External inventory

First prove access to a small, useful assortment from approved merchant feeds,
affiliate partnerships, or direct retailer agreements. Awin is one concrete route
to investigate: its publisher feeds include product links, descriptions, prices,
and images [4]. Access, merchant coverage, allowed uses, and field completeness
must be verified for the intended market. No partnership is currently assumed.

Represent at least:

- Merchant and source product ID; stable internal product and variant IDs.
- Title, category, brand, images, and optional material/color/style attributes.
- Offers with currency, price, retailer URL, size/variant stock where supplied.
- Source update time, last successful check, active state, and provenance.
- Separate permissions/provenance for display, image processing, and ML training.

Refresh on a schedule, deactivate stale or removed offers, deduplicate variants,
and track feed failures. Missing size data means unknown availability, rather
than availability in every size. Set freshness thresholds per source and expose
last-checked information where useful.

Start with coverage sufficient for the chosen audience and outfit slots. A narrow,
maintained catalog is more useful for this experiment than a large stale import.
Do not make unrestricted retailer scraping the assumed ingestion strategy.

### Outfit supervision

Prefer permissioned stylist-created outfits, partner lookbooks with usable rights,
and explicitly consented user curation. Preserve source, date, constituent items,
occasion, intended style, and judgment uncertainty.

Research datasets can support permitted experiments, but are not automatically
production data. For example, DeepFashion explicitly limits its dataset to
non-commercial research [5]. Verify the actual terms of any dataset, checkpoint,
and image source selected for this product.

Never infer that every pair in a rejected outfit is incompatible. Treat synthetic
slot replacements as potentially weak negatives until checked. Include difficult
alternatives of the same category, budget, and broadly plausible style so the
model cannot succeed only by rejecting obviously invalid combinations.

## 7. Recommendation architecture

```text
anchor garment(s) + locked outfit items + user intent
                    |
              decide outfit slots
                    |
       choose wardrobe or external inventory
                    |
       apply ownership/availability/budget constraints
                    |
          retrieve candidates for each slot
                    |
       assemble and score whole outfit alternatives
                    |
       diversify, validate, and return grounded results
```

**Retrieval** should cheaply find plausible candidates for a requested slot. Image
and text embeddings, garment attributes, category rules, and context-specific
queries can contribute. Direct similarity to the anchor is insufficient when the
target category is complementary.

**Outfit ranking** should evaluate the whole proposed set, requested occasion,
style, and preferences. Explicit constraints handle budget, required slots,
ownership, and availability; they should not depend on a model remembering them.

**Assembly** should use bounded search over per-slot shortlists, such as beam
search. Cache item representations and score batches. Do not enumerate every
possible outfit or send the whole catalog to a multimodal model.

**Diversity** should produce meaningfully different looks and avoid repeated
near-identical products. **Explanations** should use actual garment attributes
and constraints. Return inventory IDs and validated offers, never invented items.

### Experiment sequence

1. **Transparent baseline:** garment attributes, outfit templates, rules, and
   multimodal retrieval. Establish what the data and constraints alone achieve.
2. **Optional multimodal reranking experiment:** let a vision-language model compare a
   bounded shortlist of actual outfits. Measure repeatability, latency, cost,
   and blind human preference. Prefer an open model when practical; a paid API is
   an optional comparison, not a dependency of the launch plan. Treat its judgments
   as fallible, not ground truth.
3. **Learned outfit model:** train a set-aware or attention-based ranker using
   whole-outfit and preference supervision if data supports it. OutfitTransformer
   is a relevant reference for whole-outfit context and complementary retrieval,
   not a promise of performance on this product [6].

Select the simplest approach that passes the quality and operating-cost gates.
Custom training need not block a useful beta; a large model is not assumed to be
good enough without testing either.

Preserve pairwise signals as optional features, not the sole definition of
quality. Distinguish general outfit coherence from individual taste. Personalize
first through explicit preferences; learn from behavior only after enough data.

## 8. Evaluation before extensive rebuilding

Create a versioned benchmark of roughly 100–200 styling requests as a planning
starting point, not a statistically established sample requirement. Cover both
workflows, multiple photo conditions, style intents, and deliberately difficult
cases. Keep a development set separate from a frozen test set.

Compare the existing model, the transparent baseline, and each proposed replacement
using the same catalog snapshot and presentation. Record:

- Input handling: garment selection success, correction effort, crop/mask failures.
- Retrieval: relevant-candidate coverage by slot, category errors, duplicate rate.
- Outfit quality: blind preference and wearability judgments from target users;
  include multiple raters and report disagreement.
- Constraint compliance: owned-only results, budget, required slots, and verified
  stock where available.
- Product usefulness: accepted/saved outfits, swaps, explicit rejection reasons,
  time to a useful look, and return use.
- Operations: p50/p95 latency, failure rate, model cost per successful session,
  and catalog freshness.

Example provisional release gates to revise after baseline measurement:

- No known ownership or hard-budget violations in the release test cases.
- A target user finds at least one wearable look in the top three for at least
  70% of the pilot requests.
- Blind preference over the existing system is above chance with uncertainty
  reported, including results by photo type and user group.
- Expected usage fits an agreed monthly cost envelope; progress and recovery
  behavior remain usable on a phone connection.

These are proposed targets, not measured results. Do not use old pair accuracy,
an automated judge alone, or saves alone as proof of fashion quality.

For future training, split by outfit/item groups before materializing examples;
keep near-duplicate images together. Use temporal/merchant holdouts for generalization
and record seeds, source snapshots, label versions, and model artifacts.

## 9. Application architecture for launch

Use a modular monolith with an independently runnable worker:

- React/Vite frontend with TypeScript introduced as features change.
- FastAPI application with wardrobe, outfits, discovery, users, and feedback modules.
- A multiuser relational database, with PostgreSQL as the proposed launch default.
- Private object storage for uploads and generated assets.
- Background jobs for processing, feed refresh, embedding, and outfit generation.
- A replaceable retrieval implementation; retain FAISS until requirements show
  that database-integrated vectors would materially simplify operations.

Do not combine a framework rewrite, vector-store migration, and model replacement
in the first experiment. The worker is a deployment boundary, not a reason to
introduce a fleet of microservices.

For a low-cost deployment, start by profiling a small inference configuration:
batch catalog processing offline, cache garment features, and reuse them across
outfit requests. Begin with manual crop plus an affordable encoder; add automatic
segmentation only if its measured benefit warrants the compute. Evaluate
quantization and CPU inference on the actual host before assuming a continuously
running GPU is required or affordable. Keep a crop-based fallback if optional
segmentation is unavailable. Local research compute, if available, is separate
from dependable public serving capacity.

Choose hosting after measuring peak RAM, cold start, per-request CPU/GPU time,
asset volume, and background-job duration. Verify free-tier sleep behavior,
persistent storage, worker support, and quotas at selection time. A static frontend
plus one API/worker deployment can be enough for a pilot; managed infrastructure
should earn its place through reduced operating work or required capabilities.
An external product feed is a data dependency distinct from a paid inference API.

Core records: User, Garment, ImageAsset, WardrobeItem, Product, ProductVariant,
Offer, Outfit, OutfitItem, StylingRequest, RecommendationRun, FeedbackEvent,
ModelArtifact, and CatalogSource. Owned garments and retailer products may share
representations while retaining different permissions and lifecycle rules.

Replace filename-stem identities with stable IDs and separate source IDs. Store
encoder/preprocessing versions and dimensions with embeddings. Record which
inventory, model, and candidate set produced each recommendation.

Suggested additive API boundaries: garment upload/processing, wardrobe CRUD,
styling jobs and status, saved outfits, contextual slot replacement, and feedback.
Keep the current `/recommend` endpoint during migration until its replacement has
parity for required behavior. Do not force both product features into its current
flat list response.

## 10. Phased implementation and acceptance criteria

| Phase | Concrete deliverable | Exit condition |
|---|---|---|
| 0. Product and data feasibility | Audience definition, two journey sketches, catalog-source sample, evaluation rubric, cost envelope | A viable inventory path and explicit definitions of useful outfits |
| 1. Quality prototype | Real-phone photo set, crop/segmentation comparison, modern encoder comparison, transparent outfit baseline | Failure categories measured; target users can assess complete looks |
| 2. Garment and wardrobe foundation | Stable IDs, migrations, private assets, background processing, accounts, editable wardrobe | A user can add, edit, and delete items reliably; ownership isolation works |
| 3. Wardrobe styling beta | Owned-only outfits, lock/swap, save, missing-slot behavior, feedback | Agreed quality gates pass on realistic small wardrobes |
| 4. External recommendation beta | Maintained product feed, offers/filters, upload-or-wardrobe anchor, contextual recommendations | Current products and links; quality and freshness gates pass |
| 5. Public launch | Mobile polish, operational visibility, recovery, data controls, measured costs | Small external pilot succeeds and critical failures are resolved |
| 6. Learning from use | Controlled ranking experiments, improved supervision, optional personalization | Measurable quality or retention gain over the launch baseline |

Catalog feasibility work starts in phase 0 even though external discovery ships
later. If shopping is the stronger initial audience need and feed access is ready,
phases 3 and 4 can swap; both still use the shared garment foundation.

Avoid promising a calendar date before the data-access and quality experiments.
Create delivery estimates from measured prototype work and available weekly time.

## 11. Repository changes in implementation order

1. Add evaluation fixtures, a quality report format, and current baseline outputs.
2. Add garment-processing experiments outside the request path. Make encoder
   dimensions and preprocessing versions explicit.
3. Add catalog-source adapters and a current sample inventory behind a separate
   dataset/version. Do not overwrite the legacy benchmark catalog.
4. Introduce the new domain records and additive migrations; verify backups and
   import/export before switching stores.
5. Add private image assets and a job lifecycle with retries, idempotency, and status.
6. Build wardrobe entry and editing, then outfit generation and saved outfits.
7. Add context-aware slot replacement and the external discovery workflow.
8. Add quality, operational, and cost dashboards; run a limited pilot.
9. Retire the old primary model path only when replacement quality is established.

Tests should cover ownership isolation, job retry behavior, source updates,
stale-offer exclusion, outfit constraints, and deletion propagation. Preserve exact
retrieval parity tests, but separate them from semantic recommendation evaluation.

## 12. Launch essentials and feedback

Before opening the app to users: authenticated ownership checks, private uploads,
validated image dimensions/size, upload metadata handling, request and job quotas,
clear retention/deletion behavior, and backups with a tested restore path.

Provide structured error logs, job health, feed freshness, and model-run traces.
Track the funnel from upload to corrected garment, outfit viewed, saved/swapped,
and return session. Store shown candidates and positions so clicks are not treated
as unbiased preference labels. Track retailer click-through separately from
wardrobe usefulness. Financial targets require actual provider quotes and measured
usage rather than guessed monthly totals.

Run a small recruited pilot with people from the intended audience. Observe their
photos and corrections, collect outfit judgments, and investigate why they return
or stop. More supporting features should follow those observations.

## 13. Decisions still required

- Initial evaluation cohorts, styles, shopping market, and supported categories
  within the two intended audience needs.
- Whether shopping recommendations are intended to link to purchasable products.
- Which feature leads the first beta while the other remains in the release plan.
- Time available and numerical spending ceiling; open-source-first and optional
  paid model APIs are established preferences.
- Who can supply reliable current outfit judgments and usable source material.
- Whether users can try a styling request before creating an account.

None of these unresolved choices justify starting a broad code rewrite immediately.
The first work package should establish the audience, catalog feasibility, and a
small end-to-end quality experiment for both intended workflows.

## Sources and interpretation

External sources were checked on 2026-11-10. They support candidate capabilities
and research directions, not a claim that any candidate is production-ready for
FindMyFit. Repository findings above come from local source inspection.

1. [Meta SAM 3 repository](https://github.com/facebookresearch/sam3): segmentation
   candidate; evaluate checkpoint access, terms, hardware, and garment behavior.
2. [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2): reference for
   combining object grounding with segmentation.
3. [Marqo-FashionSigLIP model card](https://huggingface.co/Marqo/marqo-fashionSigLIP):
   fashion embedding candidate with product-retrieval benchmarks.
4. [Awin publisher feed guide](https://help.awin.com/developers/docs/product-feed-publisher-guide-intro):
   example of a maintained product-feed access route, not confirmed access.
5. [DeepFashion project terms](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html):
   explicit research-use restrictions; verify each selected resource separately.
6. [OutfitTransformer](https://arxiv.org/abs/2204.04812): outfit-level representations
   and target-conditioned complementary retrieval.
