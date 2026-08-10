# Technical Disclosure — Multi-Modal Signature Verification System

**Purpose of this document.** This is a technical description of the system's
design and novelty, written as a starting point for a patent attorney or a
research-paper methods section — not a patent application, and not a substitute
for one. Filing a patent is a legal process: a licensed patent attorney needs to
run a proper prior-art search, decide what is actually novel and non-obvious
under the relevant patent office's standards, and draft formal claims. Nothing
here should be represented as "patent-pending" or "patent-ready" until that
process has actually happened.

This document intentionally does **not** state a headline accuracy number. Any
number here would be a claim about system performance, and performance claims
in a patent or research disclosure need to trace to a specific, reproducible
experiment — not be asserted for effect. See [`docs/results.md`](results.md) for
every measured number this project has actually produced, with the exact
training configuration (writer count, epochs, dataset) each one came from.

## 1. Problem statement

Signature verification systems typically specialize in one capture modality:
offline systems classify a scanned image; online systems classify a stylus/touch
stroke sequence. A system that only sees one modality is blind to whatever
information the other one carries — pen pressure and timing dynamics are
invisible to an image-only classifier, and overall stroke shape/proportion is
under-used by a stroke-sequence-only classifier. Forgery detection specifically
suffers: a skilled forger can often reproduce static shape but rarely reproduces
natural pressure/velocity dynamics, and vice versa for a model given only a flat
image.

## 2. System architecture (as implemented)

```
Input (image and/or stroke capture)
  -> Region localization (YOLOv8)
  -> Preprocessing (per-modality)
  -> Static branch (Siamese CNN) ---\
  -> Dynamic branch (Transformer/LSTM) ---> Cross-attention fusion -> similarity
                                                      |
                                                      v
                                    Anomaly scoring (Isolation Forest / OC-SVM)
                                                      |
                                                      v
                                 Weighted decision fusion -> calibration -> decision
                                                      |
                                                      v
                          Explainability (Grad-CAM / attention+DTW / SHAP) -> report
```

Full diagrams: [`docs/architecture.md`](architecture.md). Code: `src/sigverify/`.

## 3. Candidate points of novelty

A patent attorney would need to run prior-art search on each of these before any
claim is drafted; they're flagged here only because they are specific, described
design choices rather than off-the-shelf combinations:

1. **Reliability-gated cross-attention fusion with graceful single-modality
   fallback** (`src/sigverify/models/fusion.py`). The two modality embeddings
   cross-attend to refine each other, then a per-modality *reliability score*
   (learned from the refined embedding itself, not a fixed weight) sets the
   softmax mixing weight. When the dynamic modality is entirely absent (a
   scanned-only signature with no stylus capture), its score is forced to
   `-inf` before the softmax, so the fused embedding degrades continuously to
   the static-only embedding rather than being corrupted by a zero-valued
   placeholder. This is a specific mechanism, not just "concatenate the two
   embeddings."
2. **Closed-loop adversarial forgery augmentation** (`src/sigverify/models/
   gan_forgery.py`, `FailureCaseBuffer`). Rather than a one-shot GAN-augmentation
   pass, the design mines the verifier's own false negatives (forgeries that
   fooled it) and false positives (genuine signatures it rejected) into a
   replay buffer that re-targets the next CycleGAN retraining cycle — the
   forgery generator is pushed toward the *current* model's specific blind
   spots, not a static forgery distribution.
3. **Stroke-level deviation scoring combining DTW alignment cost with the
   dynamic encoder's own attention weights** (`src/sigverify/explainability/
   attention_viz.py`). Most attention-based explainability reports "what the
   model looked at"; this combines that with "where the two signatures actually
   diverge in time" (via dynamic time warping) into one per-timestep score, so
   the flagged strokes are both attended-to *and* misaligned — not just one or
   the other.
4. **Decision-fusion-level SHAP attribution instead of raw-feature SHAP**
   (`src/sigverify/explainability/shap_explainer.py`). SHAP is applied to the
   small (4-input) decision-fusion function — fused similarity, static
   similarity, dynamic similarity, anomaly score — rather than to pixel or
   stroke-timestep space, producing a direct "how much did each modality drive
   this specific accept/reject call" attribution at a cost independent of image
   resolution or sequence length.
5. **Dual capture from a single signing action in the client UI**
   (`web/js/signature-pad.js`). The web console's signature pad captures the
   rendered canvas (static modality) and the raw pointer-event stroke stream —
   x, y, pressure, tilt, timestamp (dynamic modality) — from one physical
   signing gesture, rather than requiring two separate capture steps.
6. **Selectable hybrid CNN-Transformer / RNN-Transformer embedding heads**
   (`src/sigverify/models/static_branch.py`'s `HybridEmbeddingHead`,
   `src/sigverify/models/dynamic_branch.py`'s `encoder="hybrid"`). The static
   branch can route the CNN backbone's spatial feature map through a
   Transformer encoder (treating each spatial location as a token) before
   attention-pooling, instead of a plain global-average-pool head; the
   dynamic branch can route stroke sequences through a BiLSTM (local
   pen-dynamics inductive bias) followed by a Transformer (global
   self-attention across the whole stroke) instead of either alone. Both are
   selected via a constructor/CLI flag alongside the original heads, not a
   replacement — `scripts/cross_validate.py` trains both variants on
   identical writer-disjoint folds for a direct, paired comparison (see
   `docs/results.md`'s cross-validation table for the measured result, and
   `docs/research_gap.md` for why this specific hybrid pattern was chosen —
   it mirrors the CNN+Transformer approach several 2024-2026 offline
   signature verification papers report gains from).

## 4. What has and hasn't been validated

- **Validated**: every module above runs, is unit-tested (`tests/`), and has
  been exercised end-to-end on two independent real public datasets (CEDAR,
  MOBISIG — see [`data/README.md`](../data/README.md)) with results in
  [`docs/results.md`](results.md), including writer-disjoint K-fold
  cross-validation (`scripts/cross_validate.py`, mean +/- std across folds,
  not a single split) and a paired baseline-vs-hybrid architecture comparison
  on identical folds.
- **Not validated**: performance at production scale (full dataset sizes, GPU
  training, the full ResNet50/224px config in `configs/default.yaml`), or
  against modern generative (diffusion/GAN-drawn) forged signatures beyond the
  CycleGAN augmentation already built in. Cross-*dataset* generalization
  (train on CEDAR, test on a different static-image dataset) specifically has
  **not** been measured — see `docs/research_gap.md` §3 for why a second
  compatible static dataset wasn't available this round, and §2 for why that
  gap matters.
- **Not done**: any prior-art search, any claims drafting, any filing.

## 5. Suggested next steps for an actual patent or research submission

1. Run `scripts/train_static.py` / `train_dynamic.py` on the full CEDAR/GPDS and
   MOBISIG/DeepSignDB datasets with `configs/default.yaml` on a GPU, and report
   skilled-forgery and random-forgery accuracy separately (see
   [`docs/results.md`](results.md) for the exact methodology already used at
   small scale — repeat it at full scale).
2. Have a patent attorney search prior art specifically on items 1-6 above.
3. For a research submission, benchmark against at least one published
   baseline on the same dataset split (e.g. SigNet on CEDAR) rather than only
   reporting this system's own numbers — see `docs/research_gap.md` for the
   closest current published comparisons found (HTCSigNet, TransOSV,
   SignatureGuard) and why none of them report 99.5%+ either.
4. Obtain a second static-image dataset through an official (not unofficial
   mirror) channel — GPDS-960/GPDS-Synthetic or BHSig260 via their authors —
   to run the cross-dataset generalization test `docs/research_gap.md` §2
   identifies as unmeasured.
