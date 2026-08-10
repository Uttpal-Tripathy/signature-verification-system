# Research context, gap, and this project's contribution

This document exists for the same reason [`docs/technical_disclosure.md`](technical_disclosure.md)
does: to give a patent attorney or a paper's related-work section a starting
point, not to replace either. It records what was actually checked against
current (2024-2026) published literature, what gap that search surfaced, and
which parts of this repository are a direct response to it. Every citation
below was retrieved via web search/fetch during this project's development —
abstracts and search-result summaries in most cases, not full paywalled texts
(noted per entry) — so treat exact figures as "as reported by the source,"
worth re-verifying against the primary paper before citing in a formal
submission.

## 1. What the current literature reports

| Work | Year | Approach | Reported result | Access |
|---|---|---|---|---|
| HTCSigNet | 2025 | Hybrid Transformer + Convolution (SPD-Conv/SCConv CNN block + ViT block), evaluated writer-dependent (WD) and writer-independent (WI) | Multi-scale CNN+Transformer fusion improves over CNN-only baselines (exact WI numbers behind a paywall — abstract-level only) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008975) (abstract only, 403 on full text) |
| TransOSV | 2024 | Transformer-based offline signature verification | Pattern Recognition journal publication | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005800) (abstract only) |
| SignatureGuard | 2026 | Six hybrid CNN-Transformer pairings (EfficientNetB7/ResNet50 x ViT-B/16) on CEDAR (English) + ASVAR (Arabic), shared preprocessing/training pipeline | **98.2% / 98.4%** test accuracy for the best hybrid pairings | [Scientific Reports](https://www.nature.com/articles/s41598-026-62860-1) (abstract/search summary) |
| Cross-dataset generalization study | 2025 | Quantifies within-dataset vs. cross-dataset (CEDAR/GPDS/ICDAR) accuracy drop for forgery detectors | Confirms a "substantial" accuracy drop moving from within-dataset to cross-dataset evaluation | [arXiv:2510.17724](https://arxiv.org/pdf/2510.17724) (full text retrieved) |

**The headline takeaway that matters for this project's accuracy target:**
even a 2026 published hybrid CNN-Transformer paper, trained at full scale
presumably on GPU hardware, reports **98.2-98.4%** on CEDAR — not 99.5%+.
That's independent confirmation (not just this repository's own CPU-only
constraint) that 99.5%+ skilled-forgery accuracy is above what current
published state-of-the-art actually demonstrates on this class of benchmark.
See [`docs/results.md`](results.md) for why: it isn't a target this codebase
fell short of, it's a target the field hasn't demonstrated.

## 2. The research gap this project responds to

Three gaps recur across the papers above and in this project's own dataset
search (Section 3):

1. **Writer-dependent vs. writer-independent evaluation is often conflated.**
   HTCSigNet explicitly evaluates both, which is notable *because* many
   signature-verification papers report only writer-dependent numbers (test
   writers' *other* samples were in the training set) without flagging that
   this doesn't measure generalization to a brand-new enrolled user — the
   actual deployment scenario for a system like this one, where you can't
   retrain the embedding network every time someone new signs up. Every
   number in [`docs/results.md`](results.md) is writer-disjoint
   (`split_writers`/`kfold_writers` in `src/sigverify/data/datasets.py`) for
   exactly this reason.
2. **Cross-dataset generalization is measurably weaker than within-dataset
   accuracy, and rarely reported.** The 2025 study above quantifies this
   directly. This project's two independent real datasets (CEDAR for static,
   MOBISIG for dynamic) don't yet let us run a true cross-dataset test *within
   one modality* (that would need two static-image datasets, e.g. CEDAR +
   GPDS/BHSig260 — see Section 3 for why a second static dataset wasn't
   addable this round), but the multi-modal design is a direct, if partial,
   answer to the same underlying problem: a forger who can spoof one
   modality's *dataset-specific* visual quirks still has to separately spoof
   pressure/velocity dynamics, which is a different failure mode than a
   single-modality model overfitting to one dataset's scanning/capture
   artifacts.
3. **Single train/val splits are reported without variance.** With writer
   counts in the tens (25 CEDAR writers, 20 MOBISIG writers at this
   project's compute budget — see `docs/results.md`), which handful of
   writers lands in the held-out set measurably moves EER/AUC on its own.
   None of the papers above report cross-validated mean +/- std for their
   headline numbers (single train/test split is standard practice in this
   literature). `scripts/cross_validate.py` (new) runs writer-disjoint K-fold
   cross-validation and reports mean +/- std EER/AUC/accuracy across folds —
   see the cross-validation table in `docs/results.md` for the actual
   numbers this produced.

## 3. Dataset accessibility — a real, documented finding, not an excuse

Adding a third real dataset (beyond CEDAR and MOBISIG, see
[`data/README.md`](../data/README.md)) was attempted directly, not assumed
impossible:

- **SVC2004 Task 1** — confirmed genuinely open (Mendeley Data, CC BY 4.0,
  direct HTTP download via the Mendeley public API, no login required —
  verified by actually downloading it: `data.mendeley.com/datasets/dry3g9ffbt`).
  But the specific mirror found repackages the competition data as
  pre-computed ARFF feature vectors (36 engineered scalar features per
  signature), not the raw (x, y, pressure, timestamp) point sequences this
  project's dynamic branch consumes — incompatible with
  `preprocess_stroke_sequence` without changing the modeling approach
  entirely (a fixed-feature tabular classifier instead of a sequence
  encoder). The official raw-format source (cse.ust.hk/svc2004) requires an
  application/request process.
- **GPDS-960 / GPDS-Synthetic** — license agreement required directly from
  the dataset authors (gpds@gi.ulpgc.es); no open mirror found.
- **BHSig260** (Bengali + Hindi offline signatures — would have paired
  naturally with CEDAR as a second static-image dataset for the
  cross-dataset test in Section 2) — sourced from ICDAR 2009 SigComp,
  itself behind the IAPR TC11 site's access process; no CC-licensed direct
  mirror found. Some unofficial Kaggle/GitHub redistributions exist, but
  using an unofficial, unlicensed redistribution of someone else's
  competition dataset is exactly the kind of provenance problem a
  patent/paper submission shouldn't build on, so it was deliberately not
  used.

This isn't a small footnote: dataset accessibility is itself a recognized
practical barrier in this research area (most benchmark signature datasets
predate modern open-data norms and were distributed under research-only,
request-gated terms). This project's two datasets remain CEDAR and MOBISIG,
used at full or near-full writer counts (see `docs/results.md`).

## 4. What this project adds in direct response

| Gap (Section 2) | This project's response | Where |
|---|---|---|
| Writer-dependent numbers inflate accuracy | Every reported number is writer-disjoint | `split_writers`, `kfold_writers` in `src/sigverify/data/datasets.py` |
| No cross-validated variance reported | K-fold cross-validation harness, mean +/- std | `scripts/cross_validate.py`, CV table in `docs/results.md` |
| CNN-Transformer hybrids are the current SOTA direction (Section 1) but this project started CNN-only (static) / RNN-or-Transformer-only (dynamic) | Added `HybridEmbeddingHead` (CNN feature map -> Transformer self-attention over spatial tokens) for the static branch, and a BiLSTM->Transformer `encoder="hybrid"` option for the dynamic branch, evaluated head-to-head against the pre-existing baseline on identical folds | `src/sigverify/models/static_branch.py`, `src/sigverify/models/dynamic_branch.py` |
| Multi-modal fusion is under-explored as a design space (most cited work is single-modality) | Reliability-gated cross-attention fusion (`docs/technical_disclosure.md` item 1) plus a from-scratch cooperative MARL fusion experiment, reported honestly whether it wins or loses | `src/sigverify/models/fusion.py`, `notebooks/06_marl_decision_fusion.ipynb` |

None of this is presented as beating the papers in Section 1 — this
project's compute budget (CPU-only, memory-constrained, tens of writers
rather than hundreds) is far below what those papers used, and the numbers
in `docs/results.md` say so plainly. The contribution is architectural and
methodological (writer-disjoint-by-construction, cross-validated,
multi-modal, hybrid-CNN-Transformer, honestly negative results included) —
exactly the kind of thing a patent's "candidate points of novelty" or a
paper's methods section cares about, independent of whether the headline
accuracy number is state-of-the-art.
