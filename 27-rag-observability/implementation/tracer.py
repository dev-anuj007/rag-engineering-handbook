from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SpanTelemetry:
    span_id: str
    name: str
    duration_ms: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    trace_id: str
    total_duration_ms: float
    spans: list[SpanTelemetry] = field(default_factory=list)


class ObservabilityTracer:
    def __init__(self) -> None:
        self._current_trace_id = str(uuid.uuid4())[:8]
        self._spans: list[SpanTelemetry] = []
        self._start_time = time.time()

    def record_span(self, name: str, duration_ms: float, metadata: dict[str, str] | None = None) -> None:
        self._spans.append(
            SpanTelemetry(
                span_id=str(uuid.uuid4())[:6],
                name=name,
                duration_ms=duration_ms,
                metadata=metadata or {},
            )
        )

    def finalize(self) -> PipelineTrace:
        total_ms = (time.time() - self._start_time) * 1000.0
        return PipelineTrace(
            trace_id=self._current_trace_id,
            total_duration_ms=total_ms,
            spans=list(self._spans),
        )
