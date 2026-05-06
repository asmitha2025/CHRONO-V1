from dataclasses import dataclass
from typing import Optional
from .wiv_calculator import WIVResult
from .bav_calculator import BAVResult
from .icv_calculator import ICVResult

@dataclass
class MCFResult:
    mcf_score: float
    alert_level: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mcf_score": round(self.mcf_score, 4),
            "alert_level": self.alert_level,
            "band_color": "#ef4444" if self.alert_level == "RED" else "#f59e0b" if self.alert_level == "AMBER" else "#22c55e"
        }

class MCFScorer:
    RED_THRESHOLD = 0.70
    AMBER_THRESHOLD = 0.40

    def compute(self, wiv=None, bav=None, icv=None) -> MCFResult:
        if all(s is None for s in [wiv, bav, icv]):
            raise ValueError("At least one signal required.")

        weighted_score = weighted_conf = 0.0
        # Weights: WIV(0.40), BAV(0.35), ICV(0.25)
        for signal, weight in [(wiv, 0.40), (bav, 0.35), (icv, 0.25)]:
            if signal is None: continue
            norm = max(0.0, min(1.0, signal.velocity / signal.max_velocity))
            weighted_score += norm * signal.confidence * weight
            weighted_conf  += signal.confidence * weight

        mcf = weighted_score / weighted_conf if weighted_conf > 0 else 0.0
        level = ("RED" if mcf >= self.RED_THRESHOLD else
                 "AMBER" if mcf >= self.AMBER_THRESHOLD else "GREEN")
        return MCFResult(mcf_score=mcf, alert_level=level, confidence=weighted_conf)
