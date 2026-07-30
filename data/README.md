# Data

This project ships **no signature images or stroke captures** — only the code to
preprocess, train on, and evaluate them, plus a synthetic generator for
wiring/smoke-testing. Real accuracy requires real datasets.

## Directory layout

```
data/
├── processed/           # git-ignored — manifests + preprocessed samples land here
│   └── demo/            # output of scripts/generate_demo_data.py
├── raw/                 # git-ignored — put downloaded datasets here
└── samples/             # a handful of tiny, license-clean example files (tracked)
```

`data/processed/`, `data/raw/`, and generated manifests are git-ignored (see
`.gitignore`) — never commit real biometric/signature data to a public repository.

## Recommended public datasets

| Dataset | Modality | Notes |
|---|---|---|
| [CEDAR](https://cedar.buffalo.edu/NIJ/data/signatures.rar) | Static | 55 writers, 24 genuine + 24 forged each — classic small benchmark |
| [ICDAR 2011 SigComp](https://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2011_Signature_Verification_Competition_(SigComp2011)) | Static | Dutch/Chinese offline signatures |
| [GPDS-960](http://www.gpds.ulpgc.es/) | Static | 960 writers — request access from the dataset authors |
| [SVC2004](http://www.cse.ust.hk/svc2004/) | Dynamic | Task 1 (x,y only) / Task 2 (+ pressure, azimuth, altitude) |
| [MOBISIG](https://ms.sapientia.ro/~manyi/mobisig.html) | Dynamic | Finger-drawn signatures captured on a mobile touchscreen |
| [DeepSignDB](https://github.com/BiDAlab/DeepSignDB) | Static + Dynamic | Large-scale, combines several sub-datasets; closest to this project's dual-modality design |

## Building a manifest for a new dataset

The dataset classes in `sigverify.data.datasets` are manifest-driven (see the
module docstring there), so adding a new dataset means writing a short script that
walks its native file layout and emits JSON-lines records:

```jsonc
{"path": "data/raw/cedar/writer_003/genuine/007.png", "writer_id": "writer_003", "label": "genuine"}
{"path": "data/raw/cedar/writer_003/forged/012.png",  "writer_id": "writer_003", "label": "forged"}
```

`scripts/generate_demo_data.py` (via `sigverify.data.synthetic.build_demo_dataset`)
is a worked, minimal example of both generating the files and writing the manifest.

## Synthetic demo data

```bash
python scripts/generate_demo_data.py --output data/processed/demo --num-writers 12
```

Generates deterministic, per-writer parametric "handwriting" curves (both as static
images and dynamic stroke sequences) purely so every stage of the pipeline —
preprocessing, both branches, fusion, anomaly detection, calibration, explainability,
report generation — can be exercised end-to-end without downloading anything. Numbers
from a model trained only on this data are **not** meaningful accuracy claims; they
only confirm the pipeline runs.
