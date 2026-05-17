"""Offline temperature calibration for the Prophet Arena forecaster.

Fits a single scalar inverse temperature ``β`` such that

    p_calibrated = sigmoid(β · logit(p_raw))

minimizes Brier score against resolved outcomes. ``β > 1`` sharpens
the raw probabilities (underconfident); ``β < 1`` flattens them toward
0.5 (overconfident). The Prophet Arena bot maps the fitted value to
``LLM_TEMPERATURE`` (whose semantics in ``bayes.temper`` are ``1/β`` —
larger temperature = more flattening), so the persisted JSON carries
both forms.

The harness calls :func:`fit_and_save` after scoring; the next bot run
reads ``data/prophet/calibration.json`` and uses the fitted value if
``LLM_TEMPERATURE`` is not explicitly set in the environment.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.optimize import minimize_scalar


_EPS = 1e-9


def _clip01(p: float) -> float:
    return float(min(max(p, _EPS), 1.0 - _EPS))


def _logit(p: float) -> float:
    q = _clip01(p)
    return math.log(q / (1.0 - q))


def _sigmoid(x: float) -> float:
    if x >= 0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def apply_beta(p_raw: float, beta: float) -> float:
    """Temperature-scaled probability: ``sigmoid(β · logit(p_raw))``."""
    return _clip01(_sigmoid(beta * _logit(p_raw)))


@dataclass
class CalibrationRecord:
    """One observed (raw forecast, tempered forecast, outcome) triple."""

    p_raw: float
    p_tempered: float
    outcome: bool


@dataclass
class CalibrationResult:
    """Output of :meth:`TemperatureCalibrator.fit`."""

    beta: float
    llm_temperature: float  # 1/β, the form bayes.temper expects
    n_records: int
    brier_before: float
    brier_after: float
    fitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


def brier(records: list[CalibrationRecord], beta: float | None = None) -> float:
    """Brier score over records. ``beta=None`` uses ``p_tempered`` as-is."""
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        p = r.p_tempered if beta is None else apply_beta(r.p_raw, beta)
        y = 1.0 if r.outcome else 0.0
        total += (p - y) ** 2
    return total / len(records)


class TemperatureCalibrator:
    """Fit a single inverse-temperature β by minimizing Brier."""

    BETA_LOW = 0.1
    BETA_HIGH = 10.0

    def fit(self, records: list[CalibrationRecord]) -> CalibrationResult:
        if not records:
            return CalibrationResult(
                beta=1.0,
                llm_temperature=1.0,
                n_records=0,
                brier_before=0.0,
                brier_after=0.0,
            )
        before = brier(records, beta=None)

        def loss(beta: float) -> float:
            return brier(records, beta=float(beta))

        res = minimize_scalar(
            loss, bounds=(self.BETA_LOW, self.BETA_HIGH), method="bounded"
        )
        beta = float(res.x)
        after = brier(records, beta=beta)
        return CalibrationResult(
            beta=beta,
            llm_temperature=1.0 / beta if beta > 0 else 1.0,
            n_records=len(records),
            brier_before=before,
            brier_after=after,
        )


def save(result: CalibrationResult, path: Path) -> None:
    """Atomically write a calibration result to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(result.to_dict(), indent=2))
    tmp.replace(p)


def load(path: Path) -> CalibrationResult | None:
    """Read a calibration result, returning None if absent/invalid."""
    try:
        raw = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    try:
        return CalibrationResult(
            beta=float(raw["beta"]),
            llm_temperature=float(raw["llm_temperature"]),
            n_records=int(raw.get("n_records", 0)),
            brier_before=float(raw.get("brier_before", 0.0)),
            brier_after=float(raw.get("brier_after", 0.0)),
            fitted_at=str(raw.get("fitted_at", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def fit_and_save(records: list[CalibrationRecord], path: Path) -> CalibrationResult:
    """Fit β on ``records`` and persist it to ``path``."""
    result = TemperatureCalibrator().fit(records)
    save(result, path)
    return result
