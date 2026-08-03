## What does this change?

## Why

## Testing

- [ ] `pytest` passes locally
- [ ] `ruff check src scripts api tests` passes locally
- [ ] Added/updated tests for the behavior this changes
- [ ] If this affects training/inference: reported before/after numbers per the
      methodology in `docs/results.md` (skilled-forgery and random-forgery
      accuracy reported separately, exact writer/epoch/dataset counts stated)

## Checklist

- [ ] If I touched `scripts/*.py`, the executable bit is still set
      (`git diff --summary` shows no accidental `100755 -> 100644` mode changes)
- [ ] No real signature data (images, stroke captures) committed
