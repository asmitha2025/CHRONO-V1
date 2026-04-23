"""
CHRONO — Immune Collapse Velocity (ICV) Calculator
===================================================
Tracks rate of immune ratio deterioration (NLR/PLR/RAR velocity).
Scientific basis: Su J et al. (2025) 18,780 cancer patients. Sci Reports 15, 446.

Formula:
  ImmuneScore(t) = [NLR_norm × 0.35] + [PLR_norm × 0.30] + [RAR_norm × 0.35]
  ICV = Δ ImmuneScore / Δ time (months)   Alert: ICV > 0.15 units/month
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple
from .personal_baseline import PersonalBaseline


@dataclass
class ICVResult:
    immune_score_current: float
    immune_score_previous: float
    icv: float
    delta_months: float
    nlr_current: float
    plr_current: float
    rar_current: float
    nlr_z_score: float
    plr_z_score: float
    rar_z_score: float
    vas_score: float        # Vascular Anomaly Score 0–10
    alert_flag: bool
    alert_reason: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "signal": "ICV",
            "immune_score_current": round(self.immune_score_current, 4),
            "immune_score_previous": round(self.immune_score_previous, 4),
            "icv": round(self.icv, 4),
            "delta_months": round(self.delta_months, 2),
            "ratios": {"nlr": round(self.nlr_current, 3),
                       "plr": round(self.plr_current, 2),
                       "rar": round(self.rar_current, 4)},
            "z_scores": {"nlr": round(self.nlr_z_score, 3),
                         "plr": round(self.plr_z_score, 3),
                         "rar": round(self.rar_z_score, 3)},
            "vas_score": round(self.vas_score, 2),
            "alert_flag": self.alert_flag,
            "alert_reason": self.alert_reason,
            "confidence": round(self.confidence, 2),
        }


class ICVCalculator:
    NLR_WEIGHT, PLR_WEIGHT, RAR_WEIGHT = 0.35, 0.30, 0.35
    ICV_ALERT_THRESHOLD = 0.15
    Z_ALERT_THRESHOLD = 1.5

    def __init__(self, baseline: PersonalBaseline):
        self.baseline = baseline

    def _ratios(self, m: dict):
        n, l, p = m.get("neutrophils"), m.get("lymphocytes"), m.get("platelets")
        rdw, alb = m.get("rdw"), m.get("albumin")
        nlr = m.get("nlr") or (n / l if n and l and l > 0 else None)
        plr = m.get("plr") or (p / l if p and l and l > 0 else None)
        rar = m.get("rar") or (rdw / alb if rdw and alb and alb > 0 else None)
        return nlr, plr, rar

    def _immune_score(self, m: dict):
        nlr, plr, rar = self._ratios(m)
        nz = self.baseline.normalize_value("nlr", nlr) if nlr else 0.0
        pz = self.baseline.normalize_value("plr", plr) if plr else 0.0
        rz = self.baseline.normalize_value("rar", rar) if rar else 0.0
        score = nz * self.NLR_WEIGHT + pz * self.PLR_WEIGHT + rz * self.RAR_WEIGHT
        return score, (nlr or 0), (plr or 0), (rar or 0), nz, pz, rz

    def _vas(self, nlr, nz, pz, rz) -> float:
        base = abs(nz) * 0.4 + abs(pz) * 0.3 + abs(rz) * 0.3
        bonus = max(0, (nlr - 3.0) / 2.0) if nlr > 3.0 else 0
        return min(10.0, round((base + bonus) * 3.5, 2))

    def _months(self, d1: str, d2: str) -> float:
        return abs((date.fromisoformat(d2) - date.fromisoformat(d1)).days) / 30.44

    def compute(self, timepoints: List[Tuple[str, dict]]) -> Optional[ICVResult]:
        if len(timepoints) < 2:
            return None
        dp, mp = timepoints[-2]
        dc, mc = timepoints[-1]
        delta = self._months(dp, dc)
        if delta < 0.5:
            return None

        isp, *_ = self._immune_score(mp)
        isc, nlr, plr, rar, nz, pz, rz = self._immune_score(mc)
        icv = (isc - isp) / delta
        vas = self._vas(nlr, nz, pz, rz)

        reasons = []
        if icv > self.ICV_ALERT_THRESHOLD:
            reasons.append(f"ICV {icv:+.3f} units/month exceeds 0.15 threshold")
        if nlr > 3.0:
            reasons.append(f"NLR {nlr:.2f} exceeds population threshold")
        if abs(nz) > self.Z_ALERT_THRESHOLD:
            reasons.append(f"NLR {nz:+.2f}σ above personal baseline")
        if abs(rz) > self.Z_ALERT_THRESHOLD:
            reasons.append(f"RAR {rz:+.2f}σ above personal baseline")

        avail = sum(1 for x in self._ratios(mc) if x is not None)

        return ICVResult(
            immune_score_current=isc, immune_score_previous=isp,
            icv=icv, delta_months=delta,
            nlr_current=nlr, plr_current=plr, rar_current=rar,
            nlr_z_score=nz, plr_z_score=pz, rar_z_score=rz,
            vas_score=vas,
            alert_flag=bool(reasons),
            alert_reason="; ".join(reasons) if reasons else "No significant immune dysregulation",
            confidence=avail / 3.0,
        )

    def compute_full_history(self, timepoints: List[Tuple[str, dict]]) -> List[dict]:
        results = []
        for dt, m in timepoints:
            score, nlr, plr, rar, *_ = self._immune_score(m)
            results.append({"date": dt, "immune_score": round(score, 4),
                            "nlr": round(nlr, 3), "plr": round(plr, 2), "rar": round(rar, 4)})
        return results
