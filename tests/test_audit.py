from sigverify.audit.ledger import AuditLedger


def test_ledger_append_and_verify(tmp_path):
    ledger = AuditLedger(tmp_path / "audit_log.jsonl")
    ledger.append({"decision": "Genuine", "score": 0.91})
    ledger.append({"decision": "Forged", "score": 0.12})
    ledger.append({"decision": "Review", "score": 0.62})

    status = ledger.verify_chain()
    assert status["valid"] is True
    assert status["total_entries"] == 3


def test_ledger_detects_tampering(tmp_path):
    path = tmp_path / "audit_log.jsonl"
    ledger = AuditLedger(path)
    ledger.append({"decision": "Genuine", "score": 0.91})
    ledger.append({"decision": "Forged", "score": 0.12})

    # Tamper with the first entry's payload without recomputing the hash chain.
    lines = path.read_text(encoding="utf-8").splitlines()
    import json

    first = json.loads(lines[0])
    first["data"]["score"] = 0.99
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = AuditLedger(path).verify_chain()
    assert status["valid"] is False
    assert status["broken_at_index"] == 0
