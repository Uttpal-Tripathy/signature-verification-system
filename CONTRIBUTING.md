# Contributing

Thanks for considering a contribution to SIGNUM / the signature verification system.

## Getting set up

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Then confirm the baseline is green before you change anything:

```bash
pytest
ruff check src scripts api tests
```

## Making a change

1. Open an issue first for anything non-trivial (new module, architecture change,
   behavior change) so we can agree on the approach before you invest time.
2. Keep pull requests focused — one logical change per PR is much easier to review
   than a bundle of unrelated fixes.
3. Add or update tests for anything you change in `src/sigverify/`. `tests/` mirrors
   the package layout.
4. Run `pytest` and `ruff check` locally before opening the PR — CI runs both on
   Python 3.10 and 3.11.
5. If you're touching `scripts/*.py`, keep the executable bit set
   (`git update-index --chmod=+x path/to/script.py`) since these carry a
   `#!/usr/bin/env python` shebang and `ruff`'s `EXE001` rule enforces the two stay
   consistent — Windows checkouts silently drop the executable bit, so double-check
   `git diff --summary` shows no unintended mode changes before committing.

## Reporting performance numbers

If your change affects training or inference (a new architecture option, a
preprocessing change, a different config), report before/after numbers using the
same methodology as [`docs/results.md`](docs/results.md): writer-disjoint
validation split, skilled-forgery and random-forgery accuracy reported
**separately**, and the exact writer count / epoch count / dataset your numbers
came from. Don't report only the easier random-forgery number as "the" accuracy.

## Datasets

Never commit real signature images, stroke captures, or anything under
`data/raw/` or `data/processed/` — both are git-ignored for a reason. See
[`data/README.md`](data/README.md) for how to fetch the real datasets this
project's notebooks use, and `scripts/generate_demo_data.py` for the synthetic
generator used in tests and quick smoke checks.

## Code style

- `ruff check` is the source of truth; there's no separate style guide beyond
  what it enforces plus what's already in the codebase.
- No comments explaining *what* code does — only *why*, when the reason isn't
  obvious from reading it.
- Prefer editing an existing module over adding a new abstraction layer.

## Questions

Open an issue — see [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) for the
bug report / feature request templates.
