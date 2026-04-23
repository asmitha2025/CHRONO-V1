"""
CHRONO — Warburg Index Velocity (WIV) Calculator
=================================================
Measures the rate of metabolic shift toward aerobic glycolysis —
the Nobel-Prize-winning fingerprint of cancer metabolism.

Scientific basis:
  - Warburg O. (1924). Nobel Prize 1931 — cancer aerobic glycolysis
  - LDH-A elevation in aggressive cancer serum: PMC11065116 (2024)
  - RDW as inflammatory/metabolic stress marker: validated across cancer cohorts

Formula (from CHRONO Technical Document):
  Warburg_Index(t) = [LDH_norm(t) × 0.40] + [RDW_norm(t) × 0.35] + [Glucose_trend(t) × 0.25]
  WIV = Δ Warburg_Index / Δ time (months)

  Where normalization = (value - personal_mean) / personal_std
  Personal deviation > 1.5σ triggers WIV alert flag
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from .personal_baseline import PersonalBaseline


@dataclass
class WIVResult:
    """Result of a Warburg Index Velocity computation."""
    warburg_index_current: float     # WI at most recent timepoint
    warburg_index_previous: float    # WI at previous timepoint
    wiv: float                       # Rate of change (WI units / month)
    delta_months: float              # Time delta between measurements
    ldh_component: float             # LDH contribution to WI
    rdw_component: float             # RDW contribution to WI
    glucose_component: float         # Glucose trend contribution to WI
    ldh_z_score: float               # LDH deviation from personal mean
    rdw_z_score: float               # RDW deviation from personal mean
    glucose_z_score: float           # Glucose deviation from personal mean
    alert_flag: bool                 # True if WIV signal is significant
    alert_reason: str                # Human-readable reason for alert
    confidence: float                # 0.0–1.0 based on data completeness

    def to_dict(self) -> dict:
        return {
            "signal": "WIV",
            "warburg_index_current": round(self.warburg_index_current, 4),
            "warburg_index_previous": round(self.warburg_index_previous, 4),
            "wiv": round(self.wiv, 4),
            "delta_months": round(self.delta_months, 2),
            "components": {
                "ldh": round(self.ldh_component, 4),
                "rdw": round(self.rdw_component, 4),
                "glucose": round(self.glucose_component, 4),
            },
            "z_scores": {
                "ldh": round(self.ldh_z_score, 3),
                "rdw": round(self.rdw_z_score, 3),
                "glucose": round(self.glucose_z_score, 3),
            },
            "alert_flag": self.alert_flag,
            "alert_reason": self.alert_reason,
            "confidence": round(self.confidence, 2),
        }


class WIVCalculator:
    """
    Warburg Index Velocity Calculator.

    Detects the rate of metabolic shift toward aerobic glycolysis
    by tracking LDH, RDW, and Glucose trends against personal baseline.

    Usage:
        calc = WIVCalculator(baseline)
        result = calc.compute(
            timepoints=[
                ("2023-01-15", {"ldh": 168, "rdw": 13.1, "glucose": 94}),
                ("2024-01-20", {"ldh": 210, "rdw": 14.2, "glucose": 101}),
            ]
        )
    """

    # Weights from CHRONO Technical Document
    LDH_WEIGHT = 0.40
    RDW_WEIGHT = 0.35
    GLUCOSE_WEIGHT = 0.25

    # Alert threshold: personal deviation > 1.5 sigma
    ALERT_Z_THRESHOLD = 1.5
    # Minimum WIV rate to trigger alert
    WIV_ALERT_THRESHOLD = 0.05

    def __init__(self, baseline: PersonalBaseline):
        self.baseline = baseline

    def compute_warburg_index(
        self,
        ldh: Optional[float],
        rdw: Optional[float],
        glucose: Optional[float],
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute Warburg Index at a single timepoint.

        Returns:
            (warburg_index, ldh_component, rdw_component,
             glucose_component, ldh_z, rdw_z, glucose_z)
        """
        ldh_z = self.baseline.normalize_value("ldh", ldh) if ldh is not None else 0.0
        rdw_z = self.baseline.normalize_value("rdw", rdw) if rdw is not None else 0.0
        glucose_z = self.baseline.normalize_value("glucose", glucose) if glucose is not None else 0.0

        ldh_comp = ldh_z * self.LDH_WEIGHT
        rdw_comp = rdw_z * self.RDW_WEIGHT
        glucose_comp = glucose_z * self.GLUCOSE_WEIGHT

        wi = ldh_comp + rdw_comp + glucose_comp
        return wi, ldh_comp, rdw_comp, glucose_comp, ldh_z, rdw_z, glucose_z

    def _months_between(self, date1: str, date2: str) -> float:
        """Compute approximate months between two ISO date strings."""
        from datetime import date
        d1 = date.fromisoformat(date1)
        d2 = date.fromisoformat(date2)
        delta_days = abs((d2 - d1).days)
        return delta_days / 30.44  # average days per month

    def compute(
        self,
        timepoints: List[Tuple[str, dict]],
    ) -> Optional[WIVResult]:
        """
        Compute WIV across multiple timepoints.

        Args:
            timepoints: List of (date_string, marker_dict) tuples,
                        sorted chronologically. Minimum 2 timepoints required.

        Returns:
            WIVResult if computable, None if insufficient data.
        """
        if len(timepoints) < 2:
            return None

        # Use the last two timepoints
        date_prev, markers_prev = timepoints[-2]
        date_curr, markers_curr = timepoints[-1]

        delta_months = self._months_between(date_prev, date_curr)
        if delta_months < 0.5:
            return None  # Too close together to compute meaningful velocity

        # Compute Warburg Index at both timepoints
        wi_prev, ldh_cp, rdw_cp, gluc_cp, ldh_z_p, rdw_z_p, gluc_z_p = (
            self.compute_warburg_index(
                markers_prev.get("ldh"),
                markers_prev.get("rdw"),
                markers_prev.get("glucose"),
            )
        )
        wi_curr, ldh_cc, rdw_cc, gluc_cc, ldh_z_c, rdw_z_c, gluc_z_c = (
            self.compute_warburg_index(
                markers_curr.get("ldh"),
                markers_curr.get("rdw"),
                markers_curr.get("glucose"),
            )
        )

        # Velocity = change per month
        wiv = (wi_curr - wi_prev) / delta_months

        # Determine alert conditions
        alert_reasons = []
        if abs(ldh_z_c) > self.ALERT_Z_THRESHOLD:
            alert_reasons.append(f"LDH {ldh_z_c:+.2f}σ from personal baseline")
        if abs(rdw_z_c) > self.ALERT_Z_THRESHOLD:
            alert_reasons.append(f"RDW {rdw_z_c:+.2f}σ from personal baseline")
        if abs(gluc_z_c) > self.ALERT_Z_THRESHOLD:
            alert_reasons.append(f"Glucose {gluc_z_c:+.2f}σ from personal baseline")
        if abs(wiv) > self.WIV_ALERT_THRESHOLD:
            alert_reasons.append(f"Warburg Index velocity {wiv:+.3f} WI/month")

        alert_flag = len(alert_reasons) > 0

        # Confidence: based on how many markers are available
        available = sum(1 for k in ["ldh", "rdw", "glucose"] if k in markers_curr)
        confidence = available / 3.0

        return WIVResult(
            warburg_index_current=wi_curr,
            warburg_index_previous=wi_prev,
            wiv=wiv,
            delta_months=delta_months,
            ldh_component=ldh_cc,
            rdw_component=rdw_cc,
            glucose_component=gluc_cc,
            ldh_z_score=ldh_z_c,
            rdw_z_score=rdw_z_c,
            glucose_z_score=gluc_z_c,
            alert_flag=alert_flag,
            alert_reason="; ".join(alert_reasons) if alert_reasons else "No significant Warburg signal",
            confidence=confidence,
        )

    def compute_full_history(
        self, timepoints: List[Tuple[str, dict]]
    ) -> List[dict]:
        """Compute Warburg Index at every timepoint (for trend charts)."""
        results = []
        for date, markers in timepoints:
            wi, ldh_c, rdw_c, gluc_c, ldh_z, rdw_z, gluc_z = self.compute_warburg_index(
                markers.get("ldh"),
                markers.get("rdw"),
                markers.get("glucose"),
            )
            results.append({
                "date": date,
                "warburg_index": round(wi, 4),
                "ldh_z": round(ldh_z, 3),
                "rdw_z": round(rdw_z, 3),
                "glucose_z": round(gluc_z, 3),
            })
        return results
