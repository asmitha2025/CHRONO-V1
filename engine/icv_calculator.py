from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import date
from .personal_baseline import PersonalBaseline

@dataclass
class ICVResult:
    icv: float
    confidence: float
    velocity: float = 0.0 # for MCFScorer
    max_velocity: float = 0.3

    def __post_init__(self):
        self.velocity = self.icv

    def to_dict(self) -> dict:
        return {"icv": round(self.icv, 4), "confidence": round(self.confidence, 2)}

class ICVCalculator:
    """Immune Composite Velocity -- NLR/PLR/RAR dysregulation rate."""
    # Dual-threshold alert: velocity AND high personal percentile
    ALERT_ICV = 0.02         # composite-units/month
    ALERT_NLR_PCT = 85.0     # personal 85th percentile

    def __init__(self, baseline: PersonalBaseline):
        self.baseline = baseline

    def _compute_ratios(self, markers):
        n = markers.get("neutrophils")
        l = markers.get("lymphocytes")
        p = markers.get("platelets")
        rdw = markers.get("rdw")
        alb = markers.get("albumin")
        nlr = n/l if n and l and l > 0 else None
        plr = p/l if p and l and l > 0 else None
        rar = rdw/alb if rdw and alb and alb > 0 else None
        return nlr, plr, rar

    def _immune_score(self, m: dict):
        nlr, plr, rar = self._compute_ratios(m)
        nz = self.baseline.zscore("nlr", nlr) if nlr else 0.0
        pz = self.baseline.zscore("plr", plr) if plr else 0.0
        rz = self.baseline.zscore("rar", rar) if rar else 0.0
        # Weights: NLR(0.35), PLR(0.30), RAR(0.35)
        score = (nz or 0) * 0.35 + (pz or 0) * 0.30 + (rz or 0) * 0.35
        return score, nlr, nz

    def _alert(self, icv, nlr_pct):
        # Must pass BOTH thresholds -- reduces false positives
        return icv > self.ALERT_ICV and (nlr_pct or 0) > self.ALERT_NLR_PCT

    def _months(self, d1: str, d2: str) -> float:
        return abs((date.fromisoformat(d2) - date.fromisoformat(d1)).days) / 30.44

    def compute(self, timepoints: List[Tuple[str, dict]]) -> Optional[ICVResult]:
        if len(timepoints) < 2: return None
        dp, mp = timepoints[-2]
        dc, mc = timepoints[-1]
        delta = self._months(dp, dc)
        isp, *_ = self._immune_score(mp)
        isc, nlr, nz = self._immune_score(mc)
        icv = (isc - isp) / delta
        avail = sum(1 for x in self._compute_ratios(mc) if x is not None)
        return ICVResult(icv=icv, confidence=avail/3.0)
