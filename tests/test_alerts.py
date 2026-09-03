from sigverify.alerts import AlertBroker, AlertSeverity, classify_alert
from sigverify.pipeline.inference import VerificationResult


def _result(decision, combined_score=0.5, anomaly_score=None, is_novel=None):
    return VerificationResult(
        decision=decision,
        combined_score=combined_score,
        fused_similarity=0.5,
        static_similarity=0.5,
        dynamic_similarity=0.5,
        calibrated_score=0.5,
        anomaly_score=anomaly_score,
        is_novel=is_novel,
        confidence_interval=None,
        modality_weights={"static": 0.5, "dynamic": 0.5},
    )


def test_classify_alert_forged_is_critical():
    alert = classify_alert(_result("Forged", combined_score=0.2), user_id="writer_1")
    assert alert.severity == AlertSeverity.CRITICAL
    assert alert.decision == "Forged"
    assert alert.user_id == "writer_1"


def test_classify_alert_review_is_warning():
    alert = classify_alert(_result("Review", combined_score=0.6), user_id=None)
    assert alert.severity == AlertSeverity.WARNING


def test_classify_alert_genuine_is_info():
    alert = classify_alert(_result("Genuine", combined_score=0.95), user_id="writer_2")
    assert alert.severity == AlertSeverity.INFO


def test_classify_alert_genuine_but_novel_escalates_to_warning():
    alert = classify_alert(
        _result("Genuine", combined_score=0.85, anomaly_score=0.9, is_novel=True), user_id="writer_3"
    )
    assert alert.severity == AlertSeverity.WARNING
    assert "novel" in alert.message.lower()


def test_alert_ids_are_unique_and_increasing():
    a1 = classify_alert(_result("Genuine"), user_id=None)
    a2 = classify_alert(_result("Genuine"), user_id=None)
    assert a2.id > a1.id


def test_alert_to_json_roundtrips_severity_as_string():
    alert = classify_alert(_result("Forged"), user_id="writer_1")
    payload = alert.to_json()
    assert payload["severity"] == "critical"
    assert isinstance(payload["severity"], str)


def test_broker_publish_and_recent():
    broker = AlertBroker(history_size=3)
    for decision in ("Genuine", "Review", "Forged", "Genuine"):
        broker.publish(classify_alert(_result(decision), user_id=None))
    recent = broker.recent()
    assert len(recent) == 3  # bounded by history_size
    assert recent[-1].decision == "Genuine"


def test_broker_recent_respects_limit():
    broker = AlertBroker(history_size=10)
    for _ in range(5):
        broker.publish(classify_alert(_result("Genuine"), user_id=None))
    assert len(broker.recent(limit=2)) == 2


def test_broker_subscribe_receives_published_alerts():
    broker = AlertBroker()
    queue = broker.subscribe()
    alert = classify_alert(_result("Forged"), user_id="writer_9")
    broker.publish(alert)
    received = queue.get_nowait()
    assert received.id == alert.id


def test_broker_unsubscribe_stops_delivery():
    broker = AlertBroker()
    queue = broker.subscribe()
    broker.unsubscribe(queue)
    broker.publish(classify_alert(_result("Forged"), user_id=None))
    assert queue.empty()
