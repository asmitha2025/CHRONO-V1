from dataclasses import dataclass
from typing import List, Optional, Tuple
from .personal_baseline import PersonalBaseline

@dataclass
class WIVResult:
    wiv: float
    confidence: float
    velocity: float = 0.0 # for MCFScorer
    max_velocity: float = 0.5

    def __post_init__(self):
        self.velocity = self.wiv

    def to_dict(self) -> dict:
        return {"wiv": round(self.wiv, 4), "confidence": round(self.confidence, 2)}

class WIVCalculator:
    """Warburg Index Velocity -- aerobic glycolysis shift rate."""

    def __init__(self, baseline, glucose_w=0.40, ldh_w=0.30,
                 albumin_w=0.20, lactate_w=0.10):
        if abs(sum([glucose_w, ldh_w, albumin_w, lactate_w]) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        self.baseline = baseline
        self.weights = {"glucose": glucose_w, "ldh": ldh_w,
                        "albumin": albumin_w, "lactate": lactate_w}

    def _compute_wi(self, markers):
        wi, component_z, available = 0.0, {}, 0
        for marker, weight in self.weights.items():
            val = markers.get(marker)
            if val is None: continue
            z = self.baseline.zscore(marker, val) or 0.0
            # Albumin negated: low albumin = higher glycolytic shift
            contribution = -z if marker == "albumin" else z
            wi += weight * contribution
            component_z[marker] = z
            available += 1
        return wi, component_z, available

    def _months_between(self, d1: str, d2: str) -> float:
        from datetime import date
        a, b = date.fromisoformat(d1), date.fromisoformat(d2)
        return abs((b - a).days) / 30.44

    def compute(self, timepoints: List[Tuple[str, dict]]) -> Optional[WIVResult]:
        if len(timepoints) < 2: return None
        dp, mp = timepoints[-2]
        dc, mc = timepoints[-1]
        delta = self._months_between(dp, dc)
        wip, *_ = self._compute_wi(mp)
        wic, _, avail = self._compute_wi(mc)
        wiv = (wic - wip) / delta
        return WIVResult(wiv=wiv, confidence=avail/4.0)
