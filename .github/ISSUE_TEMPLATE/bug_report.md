---
name: Bug report
about: Something in the pipeline, scripts, notebooks, or web console isn't working
title: "[BUG] "
labels: bug
---

**Describe the bug**
What happened, and what did you expect instead?

**To reproduce**
Exact command / notebook cell / API request that triggers it. Include the config
file used (`configs/default.yaml`, `configs/lightweight_real.yaml`, or your own)
and, if relevant, dataset details (which manifest, how many writers).

**Traceback / logs**
Paste the full error, not just the last line.

**Environment**
- OS:
- Python version:
- `pip show torch sigverify` output (or just the versions):
- CPU-only or GPU:

**Additional context**
Anything else — e.g. if this is about a specific accuracy number, link the exact
notebook/script run and its output, per the methodology in `docs/results.md`.
