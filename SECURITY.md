# Security Policy

## Reporting a vulnerability

Please report security issues privately via
[GitHub's private vulnerability reporting](https://github.com/Uttpal-Tripathy/signature-verification-system/security/advisories/new)
rather than filing a public issue. Include what you found, how to reproduce it,
and its potential impact. We'll acknowledge reports within a reasonable time and
follow up once we've assessed severity and a fix.

## Scope and known limitations — please read before reporting

This is a research/demonstration codebase, not a hardened production biometric
system. Some things are deliberate scope decisions, not vulnerabilities to report:

- **No authentication/authorization** on the FastAPI service (`api/app.py`) — it's
  meant to sit behind whatever auth layer your deployment adds; it doesn't ship one.
- **No rate limiting** on `/api/verify` — add one at your reverse proxy/gateway if
  you deploy this publicly.
- **Model weights are not adversarially hardened.** This system has *not* been
  evaluated against adversarial perturbation attacks, model inversion, or
  membership inference. Don't treat a "Genuine" decision as forensically
  conclusive against a motivated, technically sophisticated attacker who can query
  the model — see [`docs/results.md`](docs/results.md) for its actual, measured
  accuracy before relying on it for anything with real consequences.
- **The audit ledger (`sigverify.audit.ledger`) is tamper-evident, not
  tamper-proof.** It detects after-the-fact edits to its own log file; it does not
  prevent someone with filesystem access from replacing the whole file, and the
  optional "blockchain-anchored" half described in the architecture docs is an
  unimplemented extension point, not a shipped feature.

Genuine vulnerabilities we *do* want reported: injection issues in the API/report
generation, path traversal in file handling, dependency vulnerabilities with a
real exploit path, or anything that lets a request affect data/state it shouldn't.

## Supported versions

This project doesn't yet have tagged releases with an active support/backport
policy — security fixes land on `main`.
