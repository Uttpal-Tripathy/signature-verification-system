"""Tamper-evident, hash-chained audit ledger (Gap F).

Every verification decision (and, for enrollment, every template update) is appended
as a JSON-lines record whose hash covers both its own payload and the previous
record's hash — a Merkle-style linked list. Any edit, reordering, or deletion of a
past entry breaks the hash chain from that point forward, which `verify_chain` detects.

This is a local, dependency-free implementation of the "hash-chained ledger" half of
the optional tamper-proof logging module. The "blockchain-anchored" half (periodically
publishing the chain tip hash to a public/consortium blockchain for external,
non-repudiable timestamping) is a deliberate extension point — anchoring requires
choosing and provisioning an actual chain/wallet, which is an infrastructure decision
for whoever deploys this, not something to hardcode here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def _canonical(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, timestamp: float, data: dict, algo: str = "sha256") -> str:
    payload = f"{prev_hash}|{timestamp}|{_canonical(data)}"
    return hashlib.new(algo, payload.encode("utf-8")).hexdigest()


class AuditLedger:
    def __init__(self, path: str | Path, hash_algo: str = "sha256") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hash_algo = hash_algo
        if not self.path.exists():
            self.path.touch()

    def _last_hash(self) -> str:
        last_line = None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last_line = line
        if last_line is None:
            return GENESIS_HASH
        return json.loads(last_line)["hash"]

    def append(self, data: dict[str, Any]) -> dict:
        prev_hash = self._last_hash()
        timestamp = time.time()
        entry_hash = _hash_entry(prev_hash, timestamp, data, self.hash_algo)
        entry = {"timestamp": timestamp, "prev_hash": prev_hash, "data": data, "hash": entry_hash}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        entries = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def verify_chain(self) -> dict:
        """Recomputes every entry's hash and checks linkage. Returns the first broken
        index (or None if the whole ledger checks out) plus the total entry count.
        """
        entries = self.read_all()
        expected_prev = GENESIS_HASH
        for idx, entry in enumerate(entries):
            recomputed = _hash_entry(expected_prev, entry["timestamp"], entry["data"], self.hash_algo)
            if entry["prev_hash"] != expected_prev or entry["hash"] != recomputed:
                return {"valid": False, "broken_at_index": idx, "total_entries": len(entries)}
            expected_prev = entry["hash"]
        return {"valid": True, "broken_at_index": None, "total_entries": len(entries)}
