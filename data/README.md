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

| Dataset | Modality | Access | Notes |
|---|---|---|---|
| [CEDAR](https://cedar.buffalo.edu/NIJ/data/signatures.rar) | Static | Direct download (mirror below) | 55 writers, 24 genuine + 24 skilled forgeries each — classic small benchmark |
| [ICDAR 2011 SigComp](https://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2011_Signature_Verification_Competition_(SigComp2011)) | Static | Disclaimer form on the TC11 page | Dutch/Chinese offline + online signatures |
| [GPDS-960](http://www.gpds.ulpgc.es/) | Static | Request form from the dataset authors | 960 writers |
| [SVC2004](https://cse.hkust.edu.hk/svc2004/) | Dynamic | Direct download | Task 1 (x,y only) / Task 2 (+ pressure, azimuth, altitude) |
| [MOBISIG](https://www.ms.sapientia.ro/~manyi/mobisig.html) | Dynamic | Direct download (used below) | 83 writers, finger-drawn on a capacitive touchscreen, 45 genuine + ~20 skilled forgeries each |
| [DeepSignDB](https://github.com/BiDAlab/DeepSignDB) | Static + Dynamic | Request form (BiDA Lab) | Large-scale, combines several sub-datasets; closest to this project's dual-modality design |

### Real data actually used in this repo's notebooks

Two of the datasets above need no request form or Kaggle auth — they're plain HTTP
downloads — so they're what the notebooks in [`notebooks/`](../notebooks/) train and
evaluate on:

- **CEDAR** (static) — mirrored as a single zip at
  [`nikostsagk/signature-verification`](https://github.com/nikostsagk/signature-verification/releases/download/cedar/cedar_dataset.zip)
  (254 MB; original source [cedar.buffalo.edu](https://cedar.buffalo.edu/NIJ/data/signatures.rar)).
  Free for research/demonstration use; verify licensing with the original source before
  any commercial use.
- **MOBISIG** (dynamic) — direct zip at
  [ms.sapientia.ro/~manyi/mobisig](https://www.ms.sapientia.ro/~manyi/mobisig/MOBISIG.ZIP)
  (37 MB). Per-sample CSVs with columns
  `x,y,timestamp,pressure,fingerarea,velocityx,velocityy,accelx,accely,accelz,gyrox,gyroy,gyroz`
  — no stylus tilt sensor (finger-drawn), so `tilt_x`/`tilt_y` are left at 0 rather than
  estimated. See T. Antal & L. Z. Szabó, *"Online Signature Verification on MOBISIG
  Finger-Drawn Signature Corpus,"* Mobile Information Systems, 2018.

```bash
mkdir -p data/raw/cedar data/raw/mobisig
curl -L -o data/raw/cedar/cedar_dataset.zip \
  https://github.com/nikostsagk/signature-verification/releases/download/cedar/cedar_dataset.zip
curl -L -o data/raw/mobisig/MOBISIG.ZIP \
  https://www.ms.sapientia.ro/~manyi/mobisig/MOBISIG.ZIP
unzip data/raw/cedar/cedar_dataset.zip -d data/raw/cedar/extracted
unzip data/raw/mobisig/MOBISIG.ZIP -d data/raw/mobisig/extracted

# Converts both into sigverify manifests at data/processed/real/*.jsonl
python scripts/prepare_real_datasets.py --max-writers 15   # or --full for all writers
```

`scripts/prepare_real_datasets.py` is itself a worked example of writing a manifest
converter for a new dataset with its own native file layout (CEDAR's
`original_{writer}_{n}.png` naming, MOBISIG's per-user CSV folders) — see its
docstring.

### A third dataset was investigated, not just skipped

Before settling on CEDAR + MOBISIG, a third real dataset was actively searched
for and one candidate was actually downloaded and inspected: **SVC2004 Task 1**
is genuinely open (Mendeley Data, CC BY 4.0, plain HTTP download via the
Mendeley public API, no login) — but the specific mirror repackages the raw
(x, y, pressure, timestamp) point sequences into 36 pre-computed ARFF feature
vectors per signature, which don't fit this project's sequence-based
`preprocess_stroke_sequence` pipeline without changing the modeling approach
entirely. GPDS-960/GPDS-Synthetic and BHSig260 (which would have paired with
CEDAR as a second static-image dataset) both remain behind an author
request/license process, with no CC-licensed direct mirror found. See
[`docs/research_gap.md`](../docs/research_gap.md) §3 for the full record of
what was checked.

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
