# Measured Results

Every number on this page is the direct, unedited output of a notebook or
script in this repository, run on real public data (CEDAR, MOBISIG — see
[`data/README.md`](../data/README.md)). Each entry states exactly what was
trained on and for how long, because those two things are the difference
between a proof-of-concept number and a production one. Re-running the same
notebook will reproduce results within normal small-sample and floating-point
variance, not identically.

**Two metrics, always reported separately:**
- **Skilled-forgery accuracy** — genuine vs. a deliberate forgery of *that
  writer's* signature. The hard, realistic threat model.
- **Random-forgery accuracy** — genuine vs. a genuine signature from a
  *different* writer (an impostor not even trying to forge anything). An
  easier task that scores higher; reporting only this number, unlabeled, is
  the standard way signature-verification accuracy claims get inflated.

## Static branch (Siamese CNN, CEDAR)

| Run | Writers | Epochs | Backbone / res. | Val EER (mixed) | Val AUC (mixed) | Skilled-forgery acc. | Random-forgery acc. | Notebook |
|---|---|---|---|---|---|---|---|---|
| Quick demo | 5 | 6 | MobileNetV3-L / 64px | 0.375 | 0.674 | — | — | `02_train_static_branch.ipynb` |
| High-accuracy run | 25 | 18 | MobileNetV3-L / 64px | *see below* | *see below* | *see below* | *see below* | `05_high_accuracy_training_and_evaluation.ipynb` |
| Full production config (not yet run) | 55 (all) | 50 | ResNet50 / 224px | — | — | — | — | `scripts/train_static.py` + `configs/default.yaml`, GPU recommended |

## Dynamic branch (Transformer, MOBISIG)

| Run | Writers | Epochs | Val EER (mixed) | Val AUC (mixed) | Notebook |
|---|---|---|---|---|---|
| Quick demo | 8 | 4 | 0.296 | 0.774 | `03_train_dynamic_branch.ipynb` |
| Full production config (not yet run) | 83 (all) | 50 | — | — | `scripts/train_dynamic.py` + `configs/default.yaml` |

## Why the quick-demo numbers are weak, on purpose

The quick-demo runs (5-8 writers, a handful of epochs) exist to prove the
pipeline trains correctly on real data quickly, not to be a performance claim —
and it shows: `04_fusion_and_evaluation.ipynb`, loading the quick-demo static
checkpoint, misclassified a genuine held-out skilled forgery as "Genuine"
(`combined=0.940`). That's not a bug in the pipeline; it's what an
under-trained model on 5 writers looks like, and it's the reason the
high-accuracy run (notebook 05) exists — see the section above for its actual
numbers once populated, and treat even those as a mid-scale checkpoint, not a
production result.

## What it would take to get a production-citable number

- Full writer counts (55 CEDAR, 83 MOBISIG — or a larger benchmark like
  GPDS-960 / DeepSignDB) instead of a subset.
- The full `configs/default.yaml` architecture (ResNet50 at 224px vs. this
  environment's MobileNetV3-Large at 64px, chosen only because this CPU-only
  host has under 1GB of free RAM).
- A GPU. The CPU-only host these notebooks ran on made even the "high-accuracy"
  25-writer run take on the order of tens of minutes; the full dataset at full
  resolution would take substantially longer per epoch on CPU.
- Comparison against a published baseline on the same split (e.g. SigNet's
  reported CEDAR numbers) rather than only this system's own metric.
