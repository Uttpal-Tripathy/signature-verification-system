# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project doesn't yet have tagged releases, so entries are grouped by theme
rather than version number.

## Unreleased

### Fixed
- **CI was broken since the initial commit.** `ruff`'s `EXE001` rule flagged every
  `scripts/*.py` file (shebang present, executable bit not set) on the Linux CI
  runner — Windows checkouts don't preserve the Unix executable bit, so the files
  were committed as mode 644. Fixed by setting the executable bit in the git index
  (`git update-index --chmod=+x`) for all nine scripts.
- `web/js/app.js`: `API_BASE` was computed via a tautological expression that
  always evaluated to `""`; replaced with a plain `""` and a comment.
- `web/js/signature-pad.js`: a mid-session canvas resize (responsive layout,
  window resize) redrew already-captured points at their old pixel coordinates on
  a canvas with new dimensions, visually distorting the signature. Points are now
  rescaled proportionally on resize.
- `web/index.html`: removed an unused `#padCrosshair` element with no CSS/JS
  binding.
- `web/css/theme.css`: added the `-webkit-` prefix for `backdrop-filter` (Safari).
- `ultralytics` was declared as a dependency but not installed in some
  environments, so enabling the web console's "auto-localize region" toggle threw
  an unhandled `ModuleNotFoundError` instead of a clean error.
- `sigverify.data.datasets.StaticSignatureTripletDataset` re-denoised every image
  from disk on every access — the actual bottleneck when training on real data
  (~0.4-1.6s/image vs. ~0.35s for a full forward+backward pass), not model
  compute. Added an opt-in `cache_in_memory` flag (`--cache-in-memory` on
  `scripts/train_static.py`) that preprocesses each image once.
- `notebooks/05_high_accuracy_training_and_evaluation.ipynb` saved whichever
  epoch's weights happened to run last instead of the best validation-EER
  checkpoint, so 18 epochs of overfitting on 19 writers reported worse numbers
  than the model had actually achieved mid-training. Fixed with best-checkpoint
  tracking and early stopping — see `docs/results.md` for the corrected numbers.

### Added
- Full multi-modal pipeline: YOLOv8 localization, Siamese CNN static branch,
  Transformer/LSTM dynamic branch, cross-attention fusion with reliability
  gating, CycleGAN forgery augmentation with closed-loop failure mining,
  per-writer anomaly detection, Platt/SLR calibration, Grad-CAM + attention/DTW +
  SHAP explainability, hash-chained audit ledger, PDF/JSON reporting.
- `web/`: SIGNUM — a real-time cyber-themed verification console (vanilla
  HTML/CSS/JS) served by `api/app.py`, whose signature pad captures both the
  rendered image and the raw pointer-event stroke sequence from one signing
  action.
- `notebooks/01`-`05`: train and evaluate on two real, directly-downloadable
  public datasets (CEDAR static, MOBISIG dynamic) via
  `scripts/prepare_real_datasets.py`, instead of only the synthetic generator.
- `docs/results.md`: every measured number, with skilled-forgery and
  random-forgery accuracy always reported separately.
- `docs/technical_disclosure.md`: a technical description of the system's design
  and candidate points of novelty, meant as a starting point for a patent
  attorney or research write-up — explicitly not a claim of "patent-ready" or a
  performance figure.
- Standard repo hygiene: `CONTRIBUTING.md`, `SECURITY.md`, issue/PR templates.
