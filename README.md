# Signature Verification System

A multi-modal (static image **+** dynamic stroke) forensic signature verification
pipeline: region localization → preprocessing → Siamese CNN + Transformer/LSTM
branches → cross-attention fusion → anomaly scoring → forensic calibration →
explainable, audit-logged verification reports.

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline diagrams and
the algorithm-to-literature-gap mapping this design is built against.

> **On "accuracy":** this repository is a complete, working implementation of every
> stage in the architecture — not a set of pretrained weights. Signature verification
> accuracy comes from training each branch on real signature data (see
> [`data/README.md`](data/README.md) for recommended public datasets); the included
> `scripts/generate_demo_data.py` synthetic generator exists only to prove the
> pipeline runs end-to-end, and numbers produced from it are not accuracy claims.

## What's implemented

| Stage | Approach | Code |
|---|---|---|
| Region localization | YOLOv8 (Ultralytics), full-frame fallback | `src/sigverify/localization/` |
| Preprocessing | Denoise → deskew → binarize → normalize (image); resample → z-score/min-max (stroke) | `src/sigverify/preprocessing/` |
| Static verification | Siamese CNN (ResNet50/EfficientNet-B0/MobileNetV3), contrastive + triplet loss | `src/sigverify/models/static_branch.py` |
| Dynamic verification | LSTM / GRU / Transformer encoder with attention pooling | `src/sigverify/models/dynamic_branch.py` |
| Fusion | Cross-attention + reliability-gated residual combination | `src/sigverify/models/fusion.py` |
| Forgery augmentation | CycleGAN + closed-loop failure-case mining | `src/sigverify/models/gan_forgery.py` |
| Anomaly detection | One-Class SVM / Isolation Forest (per-writer) | `src/sigverify/models/anomaly.py` |
| Calibration | Platt scaling / Score-based Likelihood Ratio | `src/sigverify/models/calibration.py` |
| Explainability | Grad-CAM (static), attention + DTW deviation (dynamic), SHAP modality split | `src/sigverify/explainability/` |
| Audit trail | Hash-chained tamper-evident ledger | `src/sigverify/audit/ledger.py` |
| Reporting | PDF + JSON forensic verification report | `src/sigverify/pipeline/report.py` |
| Serving | FastAPI `/verify` endpoint | `api/app.py` |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# CPU-only PyTorch (fastest to install; use the CUDA index for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

## Quickstart

**1. Generate a small synthetic dataset** (or point the steps below at a real
dataset's manifest — see [`data/README.md`](data/README.md)):

```bash
python scripts/generate_demo_data.py --output data/processed/demo --num-writers 20
```

**2. Train each component** (in order — fusion and calibration depend on the branch
checkpoints):

```bash
python scripts/train_static.py  --manifest data/processed/demo/static_manifest.jsonl  --output checkpoints
python scripts/train_dynamic.py --manifest data/processed/demo/dynamic_manifest.jsonl --output checkpoints
python scripts/train_fusion.py  --static-manifest data/processed/demo/static_manifest.jsonl \
                                 --dynamic-manifest data/processed/demo/dynamic_manifest.jsonl \
                                 --checkpoints checkpoints
python scripts/fit_calibration.py --static-manifest data/processed/demo/static_manifest.jsonl \
                                   --dynamic-manifest data/processed/demo/dynamic_manifest.jsonl \
                                   --checkpoints checkpoints

# Optional: GAN forgery augmentation + closed-loop retraining
python scripts/train_gan.py --manifest data/processed/demo/static_manifest.jsonl --output checkpoints --synthesize 200
python scripts/evaluate.py  --static-manifest data/processed/demo/static_manifest.jsonl --checkpoints checkpoints
```

**3. Run a verification and get a report:**

```bash
python scripts/run_inference.py \
  --reference path/to/reference_signature.png \
  --query path/to/query_signature.png \
  --reference-stroke path/to/reference_stroke.json \
  --query-stroke path/to/query_stroke.json \
  --user-id writer_003 \
  --checkpoints checkpoints \
  --output reports/case_001
```

Produces `reports/case_001.pdf` (forensic report with Grad-CAM heatmap, confidence
interval, modality contribution split) and `reports/case_001.json`.

**4. Or serve it over HTTP:**

```bash
uvicorn api.app:app --reload
curl -X POST http://127.0.0.1:8000/verify \
  -F "reference_image=@reference.png" -F "query_image=@query.png"
```

## Evaluation methodology

Training/validation splits are **writer-disjoint** (`sigverify.data.datasets.split_writers`)
so reported metrics reflect generalization to unseen writers, not memorization.
Primary metrics are Equal Error Rate (EER) and ROC-AUC (`sigverify.utils.metrics`) —
the standard benchmark metrics reported by CEDAR/GPDS/ICDAR/SVC2004/MOBISIG/DeepSignDB,
so results are directly comparable to published baselines once trained on real data.

## Project layout

```
src/sigverify/
├── preprocessing/     # image + stroke preprocessing
├── localization/       # YOLOv8 signature-region detector
├── models/             # static/dynamic branches, fusion, GAN, anomaly, calibration, losses
├── explainability/      # Grad-CAM, attention/DTW deviation, SHAP
├── data/                # manifest-driven datasets + synthetic demo generator
├── pipeline/            # end-to-end inference + report generation
└── audit/                # hash-chained audit ledger
scripts/                 # train_*.py, fit_calibration.py, evaluate.py, run_inference.py
api/                     # FastAPI service
configs/default.yaml     # all hyperparameters in one place
tests/                   # pytest suite (preprocessing, models, pipeline, audit, metrics)
```

## Testing

```bash
pytest                        # unit + integration tests (CPU, ~30s)
ruff check src scripts api tests
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
