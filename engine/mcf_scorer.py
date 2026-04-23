"""
CHRONO — MCF Scorer (Metabolic Cancer Fingerprint)
===================================================
Aggregates the three Trident signals into a single MCF score (0.0–1.0)
and determines the alert band and Protocol-99 activation.

Formula (CHRONO Technical Document):
  MCF = (WIV_norm × 0.33) + (BAV_norm × 0.40) + (ICV_norm × 0.27)

  Bands:
    0.00–0.30 : GREEN  — No significant metabolic anomaly
    0.31–0.60 : AMBER  — Monitor, increase test frequency
    0.61–0.80 : ORANGE — Protocol-99 agent activated
    0.81–1.00 : RED    — Immediate escalation to oncologist

  Weights rationale: BAV weighted highest (0.40) based on UK Biobank
  HR=1.09 per SD — strongest epidemiological backing for cancer risk.
"""

from dataclasses import dataclass
from typing import Optional
from .wiv_calculator import WIVResult
from .bav_calculator import BAVResult
from .icv_calculator import ICVResult


# MCF Band definitions
MCF_BANDS = {
    "GREEN":  (0.00, 0.30, "#22c55e", "No significant metabolic anomaly detected."),
    "AMBER":  (0.31, 0.60, "#f59e0b", "Metabolic trends warrant monitoring. Increase test frequency."),
    "ORANGE": (0.61, 0.80, "#f97316", "Protocol-99 activated. Generating triage dossier."),
    "RED":    (0.81, 1.00, "#ef4444", "Immediate escalation to oncologist required."),
}

WIV_WEIGHT = 0.33
BAV_WEIGHT = 0.40
ICV_WEIGHT = 0.27

# Normalization clamp: z-scores beyond ±3 map to score of 1.0 or 0.0
Z_CLAMP = 3.0


@dataclass
class MCFResult:
    mcf_score: float
    band: str                    # GREEN / AMBER / ORANGE / RED
    band_color: str              # Hex color for UI
    band_message: str
    protocol99_activated: bool
    wiv_score: float             # Normalized WIV contribution
    bav_score: float             # Normalized BAV contribution
    icv_score: float             # Normalized ICV contribution
    wiv_raw: float               # Raw WIV value
    bav_raw: float               # Raw BAV value
    icv_raw: float               # Raw ICV value
    wiv_alert: bool
    bav_alert: bool
    icv_alert: bool
    trident_firing: bool         # True when all three signals are alerting simultaneously
    confidence: float            # Average confidence across three signals
    summary: str                 # Human-readable one-line summary

    def to_dict(self) -> dict:
        return {
            "mcf_score": round(self.mcf_score, 4),
            "band": self.band,
            "band_color": self.band_color,
            "band_message": self.band_message,
            "protocol99_activated": self.protocol99_activated,
            "trident_firing": self.trident_firing,
            "components": {
                "wiv": {"normalized": round(self.wiv_score, 4), "raw": round(self.wiv_raw, 4), "alert": self.wiv_alert},
                "bav": {"normalized": round(self.bav_score, 4), "raw": round(self.bav_raw, 4), "alert": self.bav_alert},
                "icv": {"normalized": round(self.icv_score, 4), "raw": round(self.icv_raw, 4), "alert": self.icv_alert},
            },
            "confidence": round(self.confidence, 2),
            "summary": self.summary,
        }


def _classify_band(score: float) -> tuple:
    """Return (band_name, color, message) for a given MCF score."""
    for band, (lo, hi, color, msg) in MCF_BANDS.items():
        if lo <= score <= hi:
            return band, color, msg
    return "RED", MCF_BANDS["RED"][2], MCF_BANDS["RED"][3]


def _normalize_signal(raw_value: float, lo: float, hi: float) -> float:
    """
    Map a raw signal value to [0, 1] range using known thresholds.
    Values at or above `hi` map to 1.0; at or below `lo` map to 0.0.
    """
    if hi <= lo:
        return 0.0
    normalized = (raw_value - lo) / (hi - lo)
    return max(0.0, min(1.0, normalized))


class MCFScorer:
    """
    Metabolic Cancer Fingerprint Scorer.

    Combines WIV, BAV, and ICV results into the final MCF score.

    Usage:
        scorer = MCFScorer()
        mcf = scorer.compute(wiv_result, bav_result, icv_result)
        print(mcf.mcf_score, mcf.band, mcf.protocol99_activated)
    """

    # Normalization ranges for each signal
    # These map the raw velocity values to [0, 1]
    WIV_RANGE = (-0.5, 1.5)    # WIV in WI units/month
    BAV_RANGE = (-0.5, 1.5)    # BAV in years/month
    ICV_RANGE = (-0.3, 0.8)    # ICV in units/month

    def compute(
        self,
        wiv: Optional[WIVResult],
        bav: Optional[BAVResult],
        icv: Optional[ICVResult],
    ) -> MCFResult:
        """
        Compute the MCF score from the three Trident signal results.
        Handles missing signals gracefully (adjusts weights proportionally).
        """
        # Extract raw values and alerts
        wiv_raw = wiv.wiv if wiv else 0.0
        bav_raw = bav.bav if bav else 0.0
        icv_raw = icv.icv if icv else 0.0

        wiv_alert = wiv.alert_flag if wiv else False
        bav_alert = bav.alert_flag if bav else False
        icv_alert = icv.alert_flag if icv else False

        wiv_conf = wiv.confidence if wiv else 0.0
        bav_conf = bav.confidence if bav else 0.0
        icv_conf = icv.confidence if icv else 0.0

        # Normalize each signal to [0, 1]
        wiv_norm = _normalize_signal(wiv_raw, *self.WIV_RANGE)
        bav_norm = _normalize_signal(bav_raw, *self.BAV_RANGE)
        icv_norm = _normalize_signal(icv_raw, *self.ICV_RANGE)

        # Compute weighted MCF score
        # Adjust weights if signals are missing
        active_weights = {
            "wiv": WIV_WEIGHT if wiv else 0.0,
            "bav": BAV_WEIGHT if bav else 0.0,
            "icv": ICV_WEIGHT if icv else 0.0,
        }
        total_weight = sum(active_weights.values())
        if total_weight == 0:
            mcf_score = 0.0
        else:
            mcf_score = (
                wiv_norm * active_weights["wiv"] +
                bav_norm * active_weights["bav"] +
                icv_norm * active_weights["icv"]
            ) / total_weight

        mcf_score = round(max(0.0, min(1.0, mcf_score)), 4)

        band, color, message = _classify_band(mcf_score)
        protocol99 = mcf_score >= 0.61
        trident = wiv_alert and bav_alert and icv_alert

        avg_conf = (wiv_conf + bav_conf + icv_conf) / 3.0

        summary = (
            f"MCF {mcf_score:.2f} [{band}]. "
            f"WIV: {wiv_raw:+.3f}/mo | BAV: {bav_raw:+.3f} yrs/mo | ICV: {icv_raw:+.3f}/mo. "
            f"{'⚡ TRIDENT FIRING — all three signals co-moving.' if trident else 'Trident not fully activated.'}"
        )

        return MCFResult(
            mcf_score=mcf_score,
            band=band,
            band_color=color,
            band_message=message,
            protocol99_activated=protocol99,
            wiv_score=wiv_norm,
            bav_score=bav_norm,
            icv_score=icv_norm,
            wiv_raw=wiv_raw,
            bav_raw=bav_raw,
            icv_raw=icv_raw,
            wiv_alert=wiv_alert,
            bav_alert=bav_alert,
            icv_alert=icv_alert,
            trident_firing=trident,
            confidence=avg_conf,
            summary=summary,
        )
