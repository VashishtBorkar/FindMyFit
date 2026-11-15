# Inspect the trained models with local images

This viewer shows saved recommendations from the completed CLIP versus
FashionSigLIP experiment. It needs your local catalog images and Pillow; no
database, model checkpoints, encoder downloads, GPU, or running API is required.

The committed predictions contain 50 reproducibly sampled held-out tops that have
both trousers and shoes partners. All eight combinations are available: raw cosine,
shared contrastive, shared ranking, and two towers for each encoder. Every trained
model uses seed 42. Retrieval searches the full held-out pools of 547 trousers and
1,418 shoes. The example selector is for qualitative inspection, not another
independent test set or a substitute for the three-seed quantitative results.

## Run locally

After pulling the branch containing this viewer, activate your existing project
Python environment and run from the repository root:

```bash
python scripts/preview_model_comparison.py --images-dir "C:/path/to/your/images"
```

Replace the path with the directory containing `top`, `pants`, and `shoes` folders.
For example, `top/195783765_1.jpg` must be beneath that directory. On macOS/Linux,
use a path such as `/Users/yourname/FindMyFit/data/images` instead. If Pillow is
missing, install it with `python -m pip install Pillow`.

Open the generated **`data/model-comparison.html`** in your browser. It is a
standalone page with embedded thumbnails; no web server is needed. It defaults to
12 tops and compares FashionSigLIP shared ranking on the left with CLIP shared
ranking on the right. Use the model selectors, previous/next buttons, and top
selector to explore the results.

To include all 50 exported tops:

```bash
python scripts/preview_model_comparison.py --images-dir "C:/path/to/your/images" --queries 50
```

To choose another output file, pass `--output data/my-model-preview.html`.
Regenerate the page after changing the image directory or query count. The script
reports missing images, and placeholders show the affected items. If every image
is missing it stops with example expected paths; check the image directory and
folder layout. `--allow-missing` is available for placeholder-only diagnostics.

## What to inspect

- Compare the same top with the same method under CLIP and FashionSigLIP first.
- Inspect trousers and shoes for color, texture, silhouette, and style coherence.
- Browse several tops, including cases where either method looks worse.
- Switch methods to inspect contrastive, ranking, and two-tower differences.
- Expand **Recorded outfit partners** to see the dataset references. Green borders
  identify those partners among recommendations; other garments can also work.

The outfit preview combines each model's first-ranked trousers and first-ranked
shoes with the anchor top. These pieces are ranked independently against the top;
the model does not score the complete outfit. Scores are similarities within each
model's space and are not compatibility percentages or calibrated across models.
This viewer uses catalog items with precomputed embeddings. It does not encode
new uploads or change the main app's currently configured model.

The generated HTML contains thumbnails from your local images. It stays local
unless you share the file. Images and generated HTML are ignored by Git.

## Regenerate predictions on the training machine

```bash
python -m training.metric_learning.export_visual_comparison \
  --runs data/tracks/fallback/runs/fashionsiglip-comparison-001
```

This requires the catalog, official JSON splits, and original checkpoints. The
exporter verifies vector digests and candidate manifests against the saved runs.
The exported `predictions.json` includes portable image keys, scores, known
partners, encoder identities, and vector fingerprints, but no images or weights.

See [the experiment report](../../docs/FASHIONSIGLIP_COMPARISON.md) for metrics,
training details, and limitations.
