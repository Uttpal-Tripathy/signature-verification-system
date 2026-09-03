"""Real-time alert classification and broadcast for the verification pipeline.

Every verification decision is classified into a severity (info/warning/critical)
and published to an in-memory `AlertBroker`, which the API's `/ws/alerts` endpoint
streams to any connected Live Monitor client. This is what makes a forged or
flagged signature visible in real time to whoever is *watching* the console — not
just returned in the HTTP response to whoever happened to submit that one request.

This is intentionally a single-process, in-memory broker (an `asyncio.Queue` per
subscriber plus a bounded history ring buffer) — the right scope for one API
process serving one Live Monitor view. Fanning alerts out across multiple API
processes/machines would need a shared broker (Redis pub/sub, a message queue);
that's a deliberate extension point, not something to fake with a dependency this
project doesn't otherwise need.
"""
from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigverify.pipeline.inference import VerificationResult


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    id: int
    timestamp: float
    severity: AlertSeverity
    decision: str
    message: str
    user_id: str | None
    combined_score: float
    anomaly_score: float | None
    is_novel: bool | None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "decision": self.decision,
            "message": self.message,
            "user_id": self.user_id,
            "combined_score": self.combined_score,
            "anomaly_score": self.anomaly_score,
            "is_novel": self.is_novel,
        }


_id_counter = itertools.count(1)


def classify_alert(result: VerificationResult, user_id: str | None) -> Alert:
    """Maps a VerificationResult to a severity + human-readable message.

    `decision` ("Forged"/"Review"/"Genuine") drives severity. A novel-writer
    anomaly flag can escalate an otherwise-"Genuine" decision to warning, since
    "matches the enrolled template well enough to pass, but looks statistically
    unlike anything else this writer has produced" is exactly the kind of signal
    a human reviewer would want surfaced, not silently hidden behind a passing
    decision.
    """
    decision = result.decision
    if decision == "Forged":
        severity = AlertSeverity.CRITICAL
        message = f"Forged signature detected (score={result.combined_score:.3f})"
    elif decision == "Review":
        severity = AlertSeverity.WARNING
        message = f"Signature flagged for manual review (score={result.combined_score:.3f})"
    elif result.is_novel:
        severity = AlertSeverity.WARNING
        anomaly = result.anomaly_score if result.anomaly_score is not None else 0.0
        message = f"Genuine decision, but statistically novel for this writer (anomaly_score={anomaly:.3f})"
    else:
        severity = AlertSeverity.INFO
        message = f"Genuine signature verified (score={result.combined_score:.3f})"

    return Alert(
        id=next(_id_counter),
        timestamp=result.timestamp,
        severity=severity,
        decision=decision,
        message=message,
        user_id=user_id,
        combined_score=result.combined_score,
        anomaly_score=result.anomaly_score,
        is_novel=result.is_novel,
    )


class AlertBroker:
    """In-memory pub/sub with a bounded history ring buffer, so a client connecting
    to the live feed after some alerts already fired still gets recent context
    instead of starting from a blank screen.
    """

    def __init__(self, history_size: int = 200) -> None:
        self._history_size = history_size
        self._history: list[Alert] = []
        self._subscribers: set[asyncio.Queue] = set()

    def recent(self, limit: int | None = None) -> list[Alert]:
        if limit is None:
            return list(self._history)
        return self._history[-limit:]

    def publish(self, alert: Alert) -> None:
        self._history.append(alert)
        if len(self._history) > self._history_size:
            del self._history[: len(self._history) - self._history_size]
        for queue in list(self._subscribers):
            queue.put_nowait(alert)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)
