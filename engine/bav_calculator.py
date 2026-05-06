import math
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple
from .personal_baseline import PersonalBaseline

@dataclass
class BAVResult:
    bav: float
    baa_current: float
    chron_age_current: float
    confidence: float
    velocity: float = 0.0 # for MCFScorer
    max_velocity: float = 0.5

    def __post_init__(self):
        self.velocity = self.bav

    def to_dict(self) -> dict:
        return {
            "bav": round(self.bav, 4),
            "baa_current": round(self.baa_current, 3),
            "chron_age_current": round(self.chron_age_current, 1),
            "confidence": round(self.confidence, 2)
        }

class BAVCalculator:
    """PhenoAge-based Biological Age Velocity calculator."""

    COEFFICIENTS = {
        "albumin": -0.0336, "creatinine": 0.0095, "glucose": 0.1953,
        "crp": 0.0954,  # applied as ln(crp)
        "lymphocyte_pct": -0.0120, "mcv": 0.0268, "rdw": 0.3306,
        "alp": 0.00188, "wbc": 0.0554,
    }
    INTERCEPT = -19.907
    CHRON_AGE_COEF = 0.0804

    def __init__(self, baseline: PersonalBaseline) -> None:
        self.baseline = baseline

    def compute_phenoage(self, markers: dict, test_date: str):
        # FIX: age computed from birth_year + test_date, not a static value
        chron_age = self.baseline.chronological_age_at(test_date)
        score = self.INTERCEPT + self.CHRON_AGE_COEF * chron_age
        used = 0
        for marker, coef in self.COEFFICIENTS.items():
            val = markers.get(marker)
            if val is None: continue
            score += coef * (math.log(val) if marker == "crp" else val)
            used += 1
        return score, chron_age, used

    def _months_between(self, d1: str, d2: str) -> float:
        a, b = date.fromisoformat(d1), date.fromisoformat(d2)
        return abs((b - a).days) / 30.44

    def compute(self, timepoints: List[Tuple[str, dict]]) -> Optional[BAVResult]:
        if len(timepoints) < 2: return None
        date_p, m_p = timepoints[-2]
        date_c, m_c = timepoints[-1]
        delta = self._months_between(date_p, date_c)
        pa_p, chron_p, _ = self.compute_phenoage(m_p, date_p)
        pa_c, chron_c, used = self.compute_phenoage(m_c, date_c)
        baa_p, baa_c = pa_p - chron_p, pa_c - chron_c
        bav = (baa_c - baa_p) / delta
        return BAVResult(bav=bav, baa_current=baa_c,
            chron_age_current=chron_c, confidence=used/9.0)
            
    def compute_full_history(self, timepoints: List[Tuple[str, dict]]) -> List[dict]:
        results = []
        for dt, markers in timepoints:
            pa, chron_age, used = self.compute_phenoage(markers, dt)
            baa = pa - chron_age
            results.append({
                "date": dt,
                "phenoage": round(pa, 3),
                "baa": round(baa, 3),
                "markers_used": used,
            })
        return results
