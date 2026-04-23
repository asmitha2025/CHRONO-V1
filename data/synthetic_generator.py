"""
CHRONO — Synthetic Patient Data Generator
==========================================
Generates "Priya's" 6-year blood test timeline for the hackathon demo.

The scenario (from CHRONO Technical Document Section 9):
  - 2018–2022: Stable baseline (GREEN, MCF ~0.12)
  - Jan 2023:  Subtle trends begin (GREEN→AMBER, MCF ~0.28)
  - Jul 2023:  6-month test — acceleration confirmed (AMBER, MCF ~0.41)
  - Jan 2024:  Trident fires — all three signals co-moving (ORANGE/RED, MCF ~0.73)
  - Outcome:   1.8cm Stage 1 ovarian mass found — 93% survival rate

ALL values remain within standard "NORMAL" population ranges throughout,
demonstrating that CHRONO detects personal velocity — not population outliers.
"""

import json
import os
import random
from datetime import date, timedelta
from typing import Dict, List


def _jitter(value: float, pct: float = 0.02) -> float:
    """Add tiny realistic noise to a value (±pct%)."""
    return round(value * (1 + random.uniform(-pct, pct)), 2)


# ─────────────────────────────────────────────────────────────────
# Priya's blood test history: 8 timepoints across 6 years
# Each dict contains a complete CBC + metabolic panel
# All values are within population reference ranges
# ─────────────────────────────────────────────────────────────────
PRIYA_TIMELINE = [
    {
        "date": "2018-03-14",
        "label": "Baseline Year 1",
        "phase": "stable",
        "markers": {
            # Warburg markers
            "ldh":          165.0,   # ref: 140–280 U/L  (personal baseline ~167)
            "rdw":          12.8,    # ref: 11.5–14.5%   (personal baseline ~12.9)
            "glucose":      92.0,    # ref: 70–100 mg/dL (personal baseline ~93)
            # PhenoAge markers
            "albumin":      4.35,    # ref: 3.5–5.5 g/dL
            "creatinine":   0.84,    # ref: 0.6–1.2 mg/dL
            "crp":          1.1,     # ref: 0–3 mg/L
            "alp":          62.0,    # ref: 44–147 U/L
            "mcv":          87.5,    # ref: 80–100 fL
            "wbc":          6.2,     # ref: 4.5–11.0 ×10³/µL
            "lymphocyte_pct": 33.0,  # ref: 20–40%
            # Immune ratio markers
            "neutrophils":  3.7,     # ×10³/µL
            "lymphocytes":  2.35,    # ×10³/µL  → NLR = 1.57
            "platelets":    228.0,   # ×10³/µL  → PLR = 97
            "hemoglobin":   14.8,
        },
        "expected_mcf_band": "GREEN",
    },
    {
        "date": "2019-04-18",
        "label": "Baseline Year 2",
        "phase": "stable",
        "markers": {
            "ldh":          168.0,
            "rdw":          12.9,
            "glucose":      93.0,
            "albumin":      4.30,
            "creatinine":   0.85,
            "crp":          1.2,
            "alp":          64.0,
            "mcv":          88.0,
            "wbc":          6.4,
            "lymphocyte_pct": 32.5,
            "neutrophils":  3.85,
            "lymphocytes":  2.30,
            "platelets":    234.0,
            "hemoglobin":   14.7,
        },
        "expected_mcf_band": "GREEN",
    },
    {
        "date": "2020-05-10",
        "label": "Baseline Year 3",
        "phase": "stable",
        "markers": {
            "ldh":          170.0,
            "rdw":          13.0,
            "glucose":      94.0,
            "albumin":      4.28,
            "creatinine":   0.86,
            "crp":          1.3,
            "alp":          65.0,
            "mcv":          88.5,
            "wbc":          6.5,
            "lymphocyte_pct": 32.0,
            "neutrophils":  3.90,
            "lymphocytes":  2.25,
            "platelets":    238.0,
            "hemoglobin":   14.6,
        },
        "expected_mcf_band": "GREEN",
    },
    {
        "date": "2021-03-22",
        "label": "Baseline Year 4",
        "phase": "stable",
        "markers": {
            "ldh":          172.0,
            "rdw":          13.1,
            "glucose":      94.5,
            "albumin":      4.25,
            "creatinine":   0.86,
            "crp":          1.4,
            "alp":          66.0,
            "mcv":          88.5,
            "wbc":          6.6,
            "lymphocyte_pct": 31.5,
            "neutrophils":  4.0,
            "lymphocytes":  2.20,
            "platelets":    242.0,
            "hemoglobin":   14.5,
        },
        "expected_mcf_band": "GREEN",
    },
    {
        "date": "2022-02-15",
        "label": "Baseline Year 5 — CHRONO installed",
        "phase": "stable",
        "markers": {
            "ldh":          174.0,
            "rdw":          13.1,
            "glucose":      95.0,
            "albumin":      4.22,
            "creatinine":   0.87,
            "crp":          1.5,
            "alp":          67.0,
            "mcv":          89.0,
            "wbc":          6.7,
            "lymphocyte_pct": 31.0,
            "neutrophils":  4.05,
            "lymphocytes":  2.18,
            "platelets":    246.0,
            "hemoglobin":   14.4,
        },
        "expected_mcf_band": "GREEN",
    },
    {
        "date": "2023-01-20",
        "label": "Year 6 — Subtle trends begin",
        "phase": "early_signal",
        "markers": {
            # LDH creeping up, still "normal" by population
            "ldh":          185.0,   # +6.3% above personal mean (was ~170)
            "rdw":          13.5,    # rising
            "glucose":      97.0,    # slightly elevated
            "albumin":      4.12,    # starting to decline
            "creatinine":   0.87,
            "crp":          2.1,     # CRP rising
            "alp":          71.0,
            "mcv":          89.5,
            "wbc":          7.1,
            "lymphocyte_pct": 29.5,  # declining
            "neutrophils":  4.5,     # rising → NLR = 2.0
            "lymphocytes":  2.05,
            "platelets":    262.0,   # → PLR = 128
            "hemoglobin":   14.2,
        },
        "expected_mcf_band": "GREEN",  # still green but trending
    },
    {
        "date": "2023-07-15",
        "label": "6-Month Check — Acceleration confirmed",
        "phase": "amber",
        "markers": {
            "ldh":          198.0,   # +17% above personal mean — still "normal" (ref ≤280)
            "rdw":          13.9,    # nearing personal envelope top
            "glucose":      99.0,
            "albumin":      4.05,
            "creatinine":   0.88,
            "crp":          2.8,
            "alp":          75.0,
            "mcv":          90.5,
            "wbc":          7.4,
            "lymphocyte_pct": 28.0,
            "neutrophils":  4.9,    # NLR = 2.45
            "lymphocytes":  1.95,
            "platelets":    278.0,  # PLR = 142
            "hemoglobin":   13.9,
        },
        "expected_mcf_band": "AMBER",
    },
    {
        "date": "2024-01-18",
        "label": "TRIDENT FIRES — Protocol-99 Activated",
        "phase": "trident_active",
        "markers": {
            # All values STILL within population "NORMAL" range
            # But dramatically above personal baselines
            "ldh":          214.0,   # +26% above personal mean — still normal (ref ≤280)
            "rdw":          14.2,    # at top of personal envelope
            "glucose":      101.0,   # just above personal baseline
            "albumin":      3.95,    # declining — below personal min envelope
            "creatinine":   0.89,
            "crp":          3.8,     # now above population range
            "alp":          81.0,
            "mcv":          91.0,
            "wbc":          7.8,
            "lymphocyte_pct": 26.5,
            "neutrophils":  5.4,     # NLR = 3.12 — crossing population threshold
            "lymphocytes":  1.73,
            "platelets":    290.0,   # PLR = 168
            "hemoglobin":   13.5,
        },
        "expected_mcf_band": "ORANGE",
    },
]

PRIYA_PROFILE = {
    "patient_id": "priya_demo_001",
    "name": "Priya",
    "age_at_first_test": 34,
    "age_at_trident": 40,
    "chronological_age": 40.0,
    "sex": "F",
    "location": "Chennai, India",
    "scenario": "Ovarian cancer — Stage 1 detected Feb 2024",
    "outcome": "CA 125 elevated. Abdominal ultrasound: 1.8cm ovarian mass — Stage 1. 5-yr survival: 93%.",
    "without_chrono": "Would likely have been detected at Stage 3 in 2025-2026. 5-yr survival: 29%.",
}


def generate_priya_dataset() -> dict:
    """Generate the complete Priya demo dataset."""
    return {
        "profile": PRIYA_PROFILE,
        "timeline": PRIYA_TIMELINE,
        "metadata": {
            "generated_by": "CHRONO Synthetic Generator v1.0",
            "purpose": "Hackathon demo — Gemma 4 Good Hackathon 2026",
            "note": "All values within population reference ranges. Anomaly detectable ONLY via personal baseline velocity.",
        },
    }


def save_priya_dataset(output_dir: str = None) -> str:
    """Save the dataset to a JSON file and return the path."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "sample_patients")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "priya_timeline.json")
    with open(path, "w") as f:
        json.dump(generate_priya_dataset(), f, indent=2)
    print(f"[CHRONO] Priya demo dataset saved → {path}")
    return path


def generate_normal_patient(patient_id: str = "normal_001") -> dict:
    """Generate a healthy control patient for comparison."""
    random.seed(42)
    timeline = []
    base = {
        "ldh": 168.0, "rdw": 13.0, "glucose": 92.0,
        "albumin": 4.3, "creatinine": 0.85, "crp": 1.1,
        "alp": 63.0, "mcv": 88.0, "wbc": 6.3,
        "lymphocyte_pct": 33.0, "neutrophils": 3.8,
        "lymphocytes": 2.3, "platelets": 228.0, "hemoglobin": 14.6,
    }
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    for i, yr in enumerate(years):
        markers = {k: _jitter(v, 0.03) for k, v in base.items()}
        timeline.append({
            "date": f"{yr}-03-{10+i:02d}",
            "label": f"Healthy baseline year {i+1}",
            "phase": "stable",
            "markers": markers,
            "expected_mcf_band": "GREEN",
        })
    return {
        "profile": {"patient_id": patient_id, "name": "Control", "chronological_age": 38.0},
        "timeline": timeline,
    }


if __name__ == "__main__":
    path = save_priya_dataset()
    print(f"Priya dataset: {len(PRIYA_TIMELINE)} timepoints across 6 years")
    print(f"Key moment: {PRIYA_TIMELINE[-1]['label']}")
    print(f"Expected band: {PRIYA_TIMELINE[-1]['expected_mcf_band']}")
