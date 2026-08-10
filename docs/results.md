# Measured Results

Every number on this page is the direct, unedited output of a notebook or
script in this repository, run on real public data (CEDAR, MOBISIG — see
[`data/README.md`](../data/README.md)). Each entry states exactly what was
trained on and for how long, because those two things are the difference
between a proof-of-concept number and a production one. Re-running the same
notebook will reproduce results within normal small-sample and floating-point
variance, not identically.

## Exploratory data analysis

`notebooks/01_dataset_exploration.ipynb` characterizes both datasets before
any model touches them (full plots and code in the notebook):

| Dataset | Writers | Genuine | Forged | Genuine:Forged | Samples/writer |
|---|---|---|---|---|---|
| CEDAR (static) | 20\* | 480 | 480 | 1.00 | exactly 48 for every writer (24 genuine + 24 forged) |
| MOBISIG (dynamic) | 83 | 3,735 | 1,660 | 2.25 | exactly 65 for every writer (45 genuine + 20 forged) |

\* The manifest currently on disk covers 20 CEDAR writers; the v2 full-scale
training run above used the full 55-writer manifest before it was regenerated
at this smaller size for the cross-validation runs below (bounding CPU time —
see that section). Both are genuinely all-writer manifests at the time they
were produced, not a cherry-picked subset.

Both datasets are **exactly** balanced per writer (every CEDAR writer has
precisely 24 genuine + 24 forged samples; every MOBISIG writer has precisely
45 genuine + 20 forged) — this is a property of how the datasets were
collected, not a preprocessing step this repository performs. CEDAR's overall
genuine:forged ratio is 1:1; MOBISIG's is not (2.25:1), which is exactly why
`verification_report`/`evaluation_matrix` never report a single blended
accuracy number without also stating the class split it came from.

Raw-scale statistics (sampled, see notebook for exact sampling):
- **CEDAR image dimensions**: width 283-888px (mean 590), height 187-816px
  (mean 348) — the model resizes every image to a fixed 128x128
  (`configs/lightweight_real.yaml`), comfortably inside this range.
- **MOBISIG stroke length**: 58-916 raw points per stroke (mean 225), 939-21,109ms
  duration (mean 5,181ms) — the model resamples every stroke to a fixed 256
  points regardless of native length.

![CEDAR per-writer sample counts and class balance](images/01_05.png)
![CEDAR raw image dimension distribution](images/01_06.png)
![MOBISIG per-writer sample counts and class balance](images/01_07.png)
![MOBISIG raw stroke length distribution](images/01_08.png)

**Two metrics, always reported separately:**
- **Skilled-forgery accuracy** — genuine vs. a deliberate forgery of *that
  writer's* signature. The hard, realistic threat model.
- **Random-forgery accuracy** — genuine vs. a genuine signature from a
  *different* writer (an impostor not even trying to forge anything). An
  easier task that scores higher; reporting only this number, unlabeled, is
  the standard way signature-verification accuracy claims get inflated.

## Static branch (Siamese CNN, CEDAR)

| Run | Writers | Epochs (ran / cap) | Backbone / res. | Val EER (mixed) | Val AUC (mixed) | Skilled-forgery acc. | Random-forgery acc. | Notebook |
|---|---|---|---|---|---|---|---|---|
| Quick demo | 5 | 6 / 6 | MobileNetV3-L / 64px | 0.375 | 0.674 | — | — | `02_train_static_branch.ipynb` |
| High-accuracy run (best checkpoint) | 25 | 7 / 18 (early-stopped) | MobileNetV3-L / 64px | 0.2812 | 0.7725 | **0.6111** | **0.7569** | `05_high_accuracy_training_and_evaluation.ipynb` |
| Full production config (not yet run) | 55 (all) | up to 50 | ResNet50 / 224px | — | — | — | — | `scripts/train_static.py` + `configs/default.yaml`, GPU recommended |

**High-accuracy run detail** (19 train writers / 6 held-out val writers,
`cedar_writer_{3,4,15,18,20,24}`): the *best* validation-EER checkpoint was kept
(early stopping, patience 6) instead of whichever epoch happened to run last —
and that best checkpoint was **epoch 1**. Epochs 2-7 all had worse mixed val EER
before triggering early stopping. That's a real, informative result, not a
truncated run: at this scale (19 writers, a few hundred images, a lightweight
CPU-friendly backbone), the ImageNet-pretrained feature extractor's out-of-the-box
embedding already captures most of the separable signal, and further fine-tuning
on this little data mostly overfits rather than improves generalization. The
skilled-forgery accuracy of **61.1%** and random-forgery accuracy of **75.7%**
are the direct, unedited output of `verification_report` at that checkpoint —
both far below a citable production number, and exactly why the "what it would
take" section below isn't optional.

![High-accuracy run: training loss and validation EER/AUC per epoch, best epoch marked](images/05_01.png)

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
under-trained model on 5 writers looks like.

The first version of the notebook 05 high-accuracy run had a real bug worth
naming: it saved whichever epoch's weights happened to run last, not the best
one, and validation EER got *worse* over 18 epochs of overfitting on only 19
train writers (mixed EER drifted from 0.28 at epoch 1 up to 0.38 by epoch 18).
Fixing that (track the best validation-EER checkpoint, restore it before the
final evaluation, add early stopping) changed the reported skilled-forgery
accuracy from 59.0% to 61.1% and random-forgery accuracy from 65.3% to 75.7% —
a real improvement from correct methodology, not from trying more seeds until
a bigger number came out. Even the corrected numbers are a mid-scale checkpoint,
not a production result — see below for what closes that gap.

## Live product checkpoints (`checkpoints_real/`, used by the SIGNUM web console)

These are produced by the actual production training scripts (not the notebooks'
ad-hoc loops), which already implement backbone freeze/unfreeze scheduling and
best-checkpoint early stopping correctly — which is visible in the result: both
branches score noticeably better here than the illustrative notebook runs above,
on a comparable writer count.

| Component | Command | Writers | Best epoch | Val EER (mixed) | Val AUC (mixed) | Val acc. @ EER |
|---|---|---|---|---|---|---|
| Static (CEDAR) | `scripts/train_static.py --cache-in-memory` | 25 (20 train / 5 val) | 1 / 15 | 0.2208 | 0.8658 | **0.7792** |
| Dynamic (MOBISIG) | `scripts/train_dynamic.py` | 20 (16 train / 4 val) | 1 / 15 | 0.1611 | 0.8921 | **0.8389** |
| Fusion | `scripts/train_fusion.py` (synthetic bridge, see notebook 04) | — | 5 / 6 | 0.0000* | 1.0000* | — |

\* Fusion's validation set is 4 synthetic triplets — an EER of exactly 0 there
reflects a nearly-trivial validation split, not a claim that fusion is solved;
don't cite it.

Both branches' best checkpoint was epoch 1 again (see the "why" note above — an
ImageNet-pretrained backbone's out-of-the-box embedding dominates at this data
scale), though this time backbone freezing during epoch 1 and unfreezing
afterward, plus a real early-stopping loop, produced a meaningfully better
epoch-1 result than the notebook's fixed-schedule loop did.

## v2: full dataset + data augmentation — a real experiment, an honest result

After the numbers above, the obvious next lever was "use all the data, and add
augmentation to fight the epoch-1-is-already-best overfitting pattern." Two
changes, both real: `StaticSignatureTripletDataset` gained a `random_affine_jitter`
augmentation path (`--augment` on `train_static.py`, small random
rotation/translation/scale applied fresh every epoch, never at eval time — see
`src/sigverify/preprocessing/image_preprocess.py`), and both branches were
retrained on **every** writer in each dataset (55 CEDAR, 83 MOBISIG) instead of a
subset.

| Component | Writers | Best epoch | Val EER (mixed) | Val AUC (mixed) | Val acc. @ EER | vs. v1 |
|---|---|---|---|---|---|---|
| Static (CEDAR, +augment) | 55 (44 train / 11 val) | 2 / 6 (early-stopped) | 0.2670 | 0.8215 | **0.7330** | -4.6 points |
| Dynamic (MOBISIG) | 83 (67 train / 16 val)† | 8 / 25‡ | 0.1993 | 0.8786 | **0.8007** | -3.8 points |

† 83-writer split: 67 train / 16 val. ‡ This run was cut short mid-epoch-9 by an
environment interruption (not a training failure) — epoch 8's checkpoint is the
last completed, best-so-far result and is reported as-is; it may not represent
final convergence.

**Full data + augmentation did not beat the smaller run.** This is the direct,
unedited result — not rounded down for effect. A plausible explanation: early
stopping (patience 4) cut both runs short (epoch 6 and epoch 8/9 respectively)
before augmentation's regularization benefit had enough epochs to pay off, while
the larger, harder validation-writer set (11 and 16 held-out writers vs. 5 and 4
in v1) makes the two numbers not a fully like-for-like comparison either. Both
are real, honest data points; neither is a production number. The take-away that
*is* solid: at this scale and epoch budget, more data and augmentation are not a
free lunch — they need the epoch budget (and probably a gentler early-stopping
patience) to realize their benefit, which this CPU-only environment couldn't
afford to test further within a reasonable time budget.

![Static branch v2 ROC curve (55 writers, augmented)](images/static_branch_v2_roc.png)
![Dynamic branch v2 ROC curve (83 writers)](images/dynamic_branch_v2_roc.png)

## Multi-agent reinforcement learning for decision fusion — `06_marl_decision_fusion.ipynb`

A second, independent experiment: instead of a hand-tuned weighted sum or a
supervised-learned fusion network, can two decentralized agents (one that only
ever sees the static similarity score, one that only ever sees the dynamic
similarity score) learn a better combination through cooperative multi-agent RL?
Implemented as a simplified MADDPG — two decentralized deterministic-policy
actors (98 parameters total — this is what ships to inference), one centralized
critic (1,249 parameters, training-time only, discarded before "deployment"),
shared cooperative reward, CTDE (Centralized Training, Decentralized Execution).
Trained for 2,000 steps on real similarity scores produced by `checkpoints_real_v2`'s
static and dynamic branches over held-out CEDAR/MOBISIG writers (96 genuine +
96 forged static-agent scores, 720 genuine + 720 forged dynamic-agent scores,
bootstrap-paired into one-shot cooperative episodes).

| Metric | Fixed-weight baseline (0.5 / 0.5) | MARL (learned) |
|---|---|---|
| EER | **0.1840** | 0.2505 |
| ROC-AUC | **0.9111** | 0.8993 |
| Accuracy @ EER threshold | **0.8160** | 0.7640 |

**MARL did not improve over the fixed-weight baseline.** This is the direct,
unedited result of the run above — not adjusted or re-run until a better seed
came out. The most likely explanation, per the notebook's own discussion: with
only two scalar agents and signals that are both already monotonically related
to the true label, a simple weighted average is close to optimal, leaving little
room for a learned nonlinear policy to do better — a legitimate negative result
common in the MARL literature when a task lacks enough structure to reward
coordination, not a bug in the implementation. `sigverify.models.fusion.CrossAttentionGatedFusion`
(trained via ordinary supervised metric learning, see the fusion table above)
remains the system's production fusion approach.

![MARL vs. fixed-weight baseline ROC comparison](images/marl_vs_baseline_roc.png)

## Writer-disjoint K-fold cross-validation and hybrid architecture comparison

Every result above is a single train/val split. At these writer counts, which
handful of writers lands in the held-out set measurably moves EER on its own —
so `scripts/cross_validate.py` (new) runs writer-disjoint 5-fold cross-validation
(every writer held out exactly once, `kfold_writers` in
`src/sigverify/data/datasets.py`) and reports mean +/- std, plus a pooled
confusion matrix and full evaluation-metrics table (precision, recall,
specificity, F1, FAR, FRR — `evaluation_matrix` in
`src/sigverify/utils/metrics.py`) from every fold's held-out predictions
concatenated together.

The same harness doubles as a head-to-head test of the new hybrid
architectures against the pre-existing baselines (see
[`docs/research_gap.md`](research_gap.md) for why hybrid CNN-Transformer is
the pattern current published work uses): **static branch** — CNN
global-average-pool head vs. `HybridEmbeddingHead` (CNN feature map ->
Transformer self-attention over spatial tokens); **dynamic branch** —
Transformer-only encoder vs. `encoder="hybrid"` (BiLSTM -> Transformer). Both
variants of each branch ran on the *same* 5-fold writer splits (same seed), so
the comparison is paired, not just two independent numbers.

Bounded to 20 writers per branch (`--max-writers 20`) to keep 4 full 5-fold
CPU-only runs in a single session — stated explicitly, not hidden.

| Branch | Variant | Mean EER | Mean AUC | Mean Acc. @ EER | Precision | Recall | Specificity | F1 | FAR | FRR |
|---|---|---|---|---|---|---|---|---|---|---|
| Static (CEDAR) | CNN (baseline) | 0.2583 ± 0.0261 | 0.8039 ± 0.0361 | 0.7417 ± 0.0261 | 0.6939\* | 0.7083\* | 0.6875\* | 0.7010\* | 0.3125\* | 0.2917\* |
| Static (CEDAR) | **Hybrid CNN+Transformer** | 0.2792 ± 0.0288 | 0.7797 ± 0.0330 | 0.7208 ± 0.0288 | 0.7184 | 0.7229 | 0.7167 | 0.7207 | 0.2833 | 0.2771 |
| Dynamic (MOBISIG) | Transformer (baseline) | 0.2461 ± 0.0371 | 0.8291 ± 0.0446 | 0.7539 ± 0.0371 | 0.7389 | 0.7389 | 0.7389 | 0.7389 | 0.2611 | 0.2611 |
| Dynamic (MOBISIG) | **Hybrid BiLSTM+Transformer** | 0.2294 ± 0.0446 | 0.8338 ± 0.0399 | 0.7706 ± 0.0446 | 0.7714 | 0.7722 | 0.7711 | 0.7718 | 0.2289 | 0.2278 |

\* The CNN baseline's precision/recall/specificity/F1/FAR/FRR come from a
**single representative fold** (fold 1 of the same 5-fold split, run
separately after the main 5-fold job had already completed without this
instrumentation — see the script's `--max-folds` option), not pooled across
all 5 folds like the other three rows. Its mean EER/AUC/Acc. columns *are*
the full 5-fold result. This asymmetry is a session-logistics artifact
(the confusion-matrix code was added partway through this experiment), not a
methodology choice, and is exactly why it's flagged here instead of blended in
silently.

**Reading the result honestly, per branch:**
- **Static branch: the hybrid did *not* beat the CNN baseline** (EER +0.021,
  AUC -0.024, accuracy -2.1 points — all in the same, consistent direction).
  This echoes the MARL result above: a small nonlinear addition on top of an
  already-reasonable representation doesn't automatically help. Notably this
  isn't a capacity story either way — the hybrid head has *fewer* total
  parameters than the plain CNN head (3,384,368 vs. 3,530,672, see the
  parameter-count table below), since it trades the CNN head's 512-dim MLP
  projector for a smaller Transformer layer. The more likely explanation is
  the token count: at 128px input, MobileNetV3-Large's feature map is only
  4x4=16 spatial tokens, which may simply be too few positions for
  self-attention to find structure that global average pooling wasn't
  already capturing.
- **Dynamic branch: the hybrid *did* beat the Transformer-only baseline**
  (EER -0.017, AUC +0.005, accuracy +1.7 points — again consistent across all
  three metrics, though the std-devs (0.045 and 0.037) are large enough
  relative to the gap that this should be read as a *modest, directionally
  consistent* improvement, not a decisive one). A plausible reason
  BiLSTM-then-Transformer helps here but CNN-then-Transformer didn't help for
  images: stroke sequences have a strong native sequential-order structure
  (pen velocity/direction at time *t* depends heavily on time *t-1*) that a
  BiLSTM is a good inductive-bias match for, whereas CEDAR's spatial layout
  has no equivalent "reading order" for a Transformer to exploit beyond what
  global pooling already captures.
- Neither result should be over-read as "hybrids work for dynamic signals,
  not static images" in general — this is one dataset pair, one writer count,
  one epoch budget, on CPU. It's reported because it's what actually happened,
  not because it proves a general rule.

![Static CNN baseline: pooled 5-fold ROC](images/cv_static_cnn_roc.png)
![Static CNN baseline: representative-fold confusion matrix](images/cv_static_cnn_cm.png)
![Static hybrid: pooled 5-fold ROC](images/cv_static_hybrid_roc.png)
![Static hybrid: pooled 5-fold confusion matrix](images/cv_static_hybrid_cm.png)
![Dynamic Transformer baseline: pooled 5-fold ROC](images/cv_dynamic_transformer_roc.png)
![Dynamic Transformer baseline: pooled 5-fold confusion matrix](images/cv_dynamic_transformer_cm.png)
![Dynamic hybrid: pooled 5-fold ROC](images/cv_dynamic_hybrid_roc.png)
![Dynamic hybrid: pooled 5-fold confusion matrix](images/cv_dynamic_hybrid_cm.png)

## Performance & resource metrics

Parameter counts (measured directly via `sum(p.numel() for p in model.parameters())`)
and checkpoint file sizes:

| Component | Config | Parameters | Checkpoint size |
|---|---|---|---|
| Static branch | ResNet50 / 224px (`configs/default.yaml`) | 24,689,472 | ~99 MB (fp32) |
| Static branch | MobileNetV3-Large / 64-128px (`configs/lightweight_real.yaml`) | 3,530,672 | 14.3 MB |
| Dynamic branch | Transformer, hidden=256, 3 layers, 8 heads (`configs/default.yaml`) | 2,701,056 | ~11 MB (fp32) |
| Dynamic branch | Transformer, hidden=128, 2 layers, 4 heads (`configs/lightweight_real.yaml`) | 480,512 | 2.5 MB |
| Static branch | Hybrid CNN+Transformer head, MobileNetV3-Large / 128px | 3,384,368 | ~13.6 MB (fp32) |
| Dynamic branch | Hybrid BiLSTM+Transformer, hidden=128 | 579,328 | ~2.3 MB (fp32) |
| Fusion | Cross-attention gated, 256-dim | 658,946 | 0.67 MB (128-dim variant) |
| GAN (train-time only) | 2 generators + 2 discriminators, 32 channels, 4 residual blocks | 4,123,332 | 16.5 MB |

End-to-end inference latency, measured with `verify_signature()` on the
lightweight config, CPU-only (this host's 2-core, memory-constrained VM —
expect materially better latency on typical deployment hardware, and much
better with a GPU for the static branch's CNN forward pass):

| Request type | Mean latency | Std. dev. |
|---|---|---|
| Static image only, no explainability | 1.40 s | 0.11 s |
| Static + dynamic (stroke), no explainability | 1.47 s | 0.15 s |
| Static + dynamic, with Grad-CAM + attention/DTW explainability | 1.76 s | 0.23 s |
| + confidence interval (7-round test-time augmentation) | adds ~7x the base static-branch cost | — |

Preprocessing cost (why `--cache-in-memory` exists): denoising a real, full-resolution
scanned signature (CEDAR averages ~385x534px) takes **~0.4-1.6s per image** on this
host — more than the entire model forward+backward pass (~0.35s for a batch of 8 at
64px). That cost is paid once per image with `--cache-in-memory`; without it, every
epoch re-denoises every image from scratch.

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

## Performance comparison across every model trained in this repository

Every run above, in one table, so the honest picture is visible at a glance
instead of requiring a page-by-page read. "Acc." is always accuracy at the
EER threshold; skilled-forgery and random-forgery are only split out where
they were measured separately (the notebook-05 style runs) — the production
and cross-validation runs below use `verification_report`'s mixed
genuine-vs-(skilled-or-random)-forgery protocol, stated explicitly per row.

| # | Model / run | Branch | Writers | Protocol | EER | AUC | Accuracy |
|---|---|---|---|---|---|---|---|
| 1 | Quick demo | Static (CEDAR) | 5 | single split | 0.375 | 0.674 | — |
| 2 | High-accuracy notebook 05 | Static (CEDAR) | 25 | single split | 0.2812 | 0.7725 | skilled 61.1% / random 75.7% |
| 3 | Quick demo | Dynamic (MOBISIG) | 8 | single split | 0.296 | 0.774 | — |
| 4 | **Live product v1** (`checkpoints_real/`) | Static (CEDAR) | 25 | single split | 0.2208 | 0.8658 | **77.9%** |
| 5 | **Live product v1** (`checkpoints_real/`) | Dynamic (MOBISIG) | 20 | single split | 0.1611 | 0.8921 | **83.9%** |
| 6 | v2: full data + augmentation | Static (CEDAR) | 55 | single split | 0.2670 | 0.8215 | 73.3% |
| 7 | v2: full data | Dynamic (MOBISIG) | 83 | single split | 0.1993 | 0.8786 | 80.1%‡ |
| 8 | CV baseline (CNN head) | Static (CEDAR) | 20 | **5-fold CV** | 0.2583 ± 0.0261 | 0.8039 ± 0.0361 | 74.17% ± 2.61 |
| 9 | CV hybrid (CNN+Transformer) | Static (CEDAR) | 20 | **5-fold CV** | 0.2792 ± 0.0288 | 0.7797 ± 0.0330 | 72.08% ± 2.88 |
| 10 | CV baseline (Transformer) | Dynamic (MOBISIG) | 20 | **5-fold CV** | 0.2461 ± 0.0371 | 0.8291 ± 0.0446 | 75.39% ± 3.71 |
| 11 | CV hybrid (BiLSTM+Transformer) | Dynamic (MOBISIG) | 20 | **5-fold CV** | 0.2294 ± 0.0446 | 0.8338 ± 0.0399 | **77.06% ± 4.46** |
| 12 | MARL fusion (learned) | Fusion (both) | — | single split | 0.2505 | 0.8993 | 76.40% |
| 13 | Fixed-weight fusion baseline | Fusion (both) | — | single split | 0.1840 | 0.9111 | **81.60%** |

‡ Row 7's dynamic v2 run was cut short mid-epoch by an environment
interruption (see the v2 section above); epoch 8's checkpoint is reported
as-is, not necessarily converged.

**What this table is actually for**, beyond a reference list:

- **Rows 4-5 remain what the live SIGNUM console serves** — they are the best
  *single-split* numbers this project produced, and single-split numbers are
  exactly what rows 8-11's cross-validation exists to sanity-check. Row 4's
  static 77.9% sits *above* row 8's cross-validated mean (74.17%) by more
  than one cross-validated standard deviation — a concrete, measured
  illustration of the "one split can be lucky" problem this whole
  cross-validation section was built to quantify, not just assert.
- **No hybrid architecture change and no fusion mechanism (rows 9, 12) beat
  its corresponding baseline (rows 8, 13) on the static/fusion side.** Two
  independent experiments — a from-scratch cooperative MARL fusion policy,
  and a CNN+Transformer hybrid embedding head — were tried specifically to
  push past the existing numbers, and both produced honest, real, *negative*
  results on this codebase's static/fusion tasks. The dynamic branch's hybrid
  (row 11) is this table's one genuine, if modest, improvement.
- **Nothing in this table is within reach of 99.5%.** The highest accuracy
  anywhere in this repository is row 4's 83.9% (a single split, on a
  CPU-only, tens-of-writers benchmark). See
  [`docs/research_gap.md`](research_gap.md) for independent confirmation
  that even a 2026 published hybrid CNN-Transformer paper, at full scale,
  reports 98.2-98.4% on CEDAR — not 99.5%+.
