"""
CHRONO — Biological Age Velocity (BAV) Calculator
==================================================
Measures the rate at which biological age diverges from chronological age.

Scientific basis:
  - Levine ME et al. (2018). PhenoAge algorithm. Aging, 10(4), 573–591.
  - UK Biobank n=308,156: HR=1.09 per 1 SD BAV increment for cancer incidence
  - Breast cancer patients show +3.73 years accelerated biological age at diagnosis

Formula (PhenoAge, Levine 2018):
  PhenoAge(t) = −19.907 − (0.0336×Albumin) + (0.0095×Creatinine)
              + (0.1953×Glucose) + (0.0954×ln(CRP)) + (−0.0120×Lymphocyte%)
              + (0.0268×MCV) + (0.3306×RDW) + (0.00188×ALP)
              + (0.0554×WBC) + (0.0804×ChronologicalAge)

  BAA(t)  = PhenoAge(t) − ChronologicalAge(t)
  BAV     = Δ BAA / Δ time (months)
  Alert   : BAV > +0.3 years/month
"""

import math
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple
from .personal_baseline import PersonalBaseline


@dataclass
class BAVResult:
    phenoage_current: float      # Biological age at latest timepoint
    phenoage_previous: float     # Biological age at previous timepoint
    baa_current: float           # Biological Age Acceleration (PhenoAge − Chron. Age)
    baa_previous: float
    bav: float                   # Rate of BAA change (years/month)
    delta_months: float
    chronological_age: float
    markers_used: int            # How many of 9 PhenoAge markers were available
    alert_flag: bool             # BAV > 0.3 years/month
    alert_reason: str
    confidence: float            # markers_used / 9

    def to_dict(self) -> dict:
        return {
            "signal": "BAV",
            "phenoage_current": round(self.phenoage_current, 3),
            "phenoage_previous": round(self.phenoage_previous, 3),
            "baa_current": round(self.baa_current, 3),
            "baa_previous": round(self.baa_previous, 3),
            "bav": round(self.bav, 4),
            "delta_months": round(self.delta_months, 2),
            "chronological_age": round(self.chronological_age, 1),
            "markers_used": self.markers_used,
            "alert_flag": self.alert_flag,
            "alert_reason": self.alert_reason,
            "confidence": round(self.confidence, 2),
        }


class BAVCalculator:
    """
    Biological Age Velocity Calculator using the PhenoAge algorithm.

    Usage:
        calc = BAVCalculator(baseline, chronological_age=38.0)
        result = calc.compute([
            ("2023-01-15", {"albumin": 4.3, "creatinine": 0.85, "glucose": 94,
                            "crp": 1.2, "lymphocyte_pct": 32, "mcv": 88,
                            "rdw": 13.1, "alp": 65, "wbc": 6.5}),
            ("2024-01-20", {"albumin": 4.0, "creatinine": 0.88, "glucose": 98,
                            "crp": 2.8, "lymphocyte_pct": 27, "mcv": 91,
                            "rdw": 14.1, "alp": 78, "wbc": 7.2}),
        ])
    """

    # PhenoAge coefficients (Levine et al. 2018)
    COEFFICIENTS = {
        "albumin":       -0.0336,
        "creatinine":     0.0095,
        "glucose":        0.1953,
        "crp_log":        0.0954,   # applied to ln(CRP)
        "lymphocyte_pct": -0.0120,
        "mcv":            0.0268,
        "rdw":            0.3306,
        "alp":            0.00188,
        "wbc":            0.0554,
        "chronological_age": 0.0804,
    }
    INTERCEPT = -19.907
    BAV_ALERT_THRESHOLD = 0.3  # years/month

    def __init__(self, baseline: PersonalBaseline, chronological_age: float):
        self.baseline = baseline
        self.chronological_age = chronological_age

    def compute_phenoage(self, markers: dict, chron_age: float) -> Tuple[float, int]:
        """
        Compute PhenoAge from marker dictionary.
        Returns (phenoage_value, num_markers_used).
        Missing markers are substituted with population means from baseline.
        """
        score = self.INTERCEPT

        used = 0
        for key, coef in self.COEFFICIENTS.items():
            if key == "chronological_age":
                score += coef * chron_age
                continue
            if key == "crp_log":
                crp = markers.get("crp")
                if crp is not None and crp > 0:
                    score += coef * math.log(crp)
                    used += 1
                else:
                    print("[CHRONO Warning] CRP missing or invalid; PhenoAge confidence will be severely penalized.")
            else:
                val = markers.get(key)
                if val is not None:
                    score += coef * val
                    used += 1

        return score, used

    def _months_between(self, d1: str, d2: str) -> float:
        a, b = date.fromisoformat(d1), date.fromisoformat(d2)
        return abs((b - a).days) / 30.44

    def compute(self, timepoints: List[Tuple[str, dict]]) -> Optional[BAVResult]:
        """Compute BAV from chronologically-sorted list of (date, markers) tuples."""
        if len(timepoints) < 2:
            return None

        date_prev, markers_prev = timepoints[-2]
        date_curr, markers_curr = timepoints[-1]

        delta_months = self._months_between(date_prev, date_curr)
        if delta_months < 0.5:
            return None

        pa_prev, used_prev = self.compute_phenoage(markers_prev, self.chronological_age)
        pa_curr, used_curr = self.compute_phenoage(markers_curr, self.chronological_age)

        baa_prev = pa_prev - self.chronological_age
        baa_curr = pa_curr - self.chronological_age
        bav = (baa_curr - baa_prev) / delta_months

        alert_flag = bav > self.BAV_ALERT_THRESHOLD
        alert_reason = (
            f"Biological age accelerating at {bav:+.3f} years/month "
            f"(BAA: {baa_curr:+.2f} years above chronological age)"
            if alert_flag
            else f"BAV within normal range ({bav:+.3f} years/month)"
        )

        confidence = used_curr / 9.0
        if markers_curr.get("crp") is None or markers_curr.get("crp") <= 0:
            confidence = max(0.0, confidence - 0.25)

        return BAVResult(
            phenoage_current=pa_curr,
            phenoage_previous=pa_prev,
            baa_current=baa_curr,
            baa_previous=baa_prev,
            bav=bav,
            delta_months=delta_months,
            chronological_age=self.chronological_age,
            markers_used=used_curr,
            alert_flag=alert_flag,
            alert_reason=alert_reason,
            confidence=confidence,
        )

    def compute_full_history(self, timepoints: List[Tuple[str, dict]]) -> List[dict]:
        """Compute PhenoAge at every timepoint (for trend charts)."""
        results = []
        for dt, markers in timepoints:
            pa, used = self.compute_phenoage(markers, self.chronological_age)
            baa = pa - self.chronological_age
            results.append({
                "date": dt,
                "phenoage": round(pa, 3),
                "baa": round(baa, 3),
                "markers_used": used,
            })
        return results
