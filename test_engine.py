"""
CHRONO -- End-to-End Engine Test
Runs the full Trident Signal pipeline on Priya's demo data.
Expected output: MCF 0.65-0.75 ORANGE at Jan 2024 timepoint.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.personal_baseline import PersonalBaseline
from engine.wiv_calculator import WIVCalculator
from engine.bav_calculator import BAVCalculator
from engine.icv_calculator import ICVCalculator
from engine.mcf_scorer import MCFScorer
from data.synthetic_generator import PRIYA_TIMELINE, PRIYA_PROFILE

def run_pipeline():
    print("=" * 60)
    print("CHRONO — Trident Signal Engine Test")
    print("Patient: Priya | 6-Year Blood Test History")
    print("=" * 60)

    # 1. Build personal baseline
    baseline = PersonalBaseline(
        patient_id=PRIYA_PROFILE["patient_id"],
        chronological_age=PRIYA_PROFILE["chronological_age"],
    )
    for tp in PRIYA_TIMELINE:
        baseline.add_test(tp["date"], tp["markers"])

    print(f"\n✓ Personal baseline built from {len(PRIYA_TIMELINE)} timepoints")
    cal = baseline.get_calibration_status()
    calibrated = [k for k, v in cal.items() if v]
    print(f"✓ Calibrated markers: {len(calibrated)} → {calibrated[:6]}...")

    # 2. Build timepoints list
    timepoints = [(tp["date"], tp["markers"]) for tp in PRIYA_TIMELINE]

    # 3. Initialise calculators
    wiv_calc = WIVCalculator(baseline)
    bav_calc = BAVCalculator(baseline, chronological_age=PRIYA_PROFILE["chronological_age"])
    icv_calc = ICVCalculator(baseline)
    scorer   = MCFScorer()

    # 4. Compute signals at each timepoint pair
    print("\n─── WIV History (Warburg Index Velocity) ───")
    wiv_history = wiv_calc.compute_full_history(timepoints)
    for r in wiv_history:
        print(f"  {r['date']}: WI={r['warburg_index']:+.4f}")

    print("\n─── BAV History (Biological Age Velocity) ───")
    bav_history = bav_calc.compute_full_history(timepoints)
    for r in bav_history:
        print(f"  {r['date']}: PhenoAge={r['phenoage']:.2f} | BAA={r['baa']:+.3f} yrs")

    print("\n─── ICV History (Immune Collapse Velocity) ───")
    icv_history = icv_calc.compute_full_history(timepoints)
    for r in icv_history:
        print(f"  {r['date']}: ImmuneScore={r['immune_score']:+.4f} | NLR={r['nlr']:.2f}")

    # 5. Final Trident Signal (latest two timepoints = Jul23 → Jan24)
    print("\n─── TRIDENT SIGNAL — Jan 2024 (Critical Timepoint) ───")
    wiv_result = wiv_calc.compute(timepoints)
    bav_result = bav_calc.compute(timepoints)
    icv_result = icv_calc.compute(timepoints)
    mcf_result = scorer.compute(wiv_result, bav_result, icv_result)

    print(f"\n  WIV: {wiv_result.wiv:+.4f} WI/month | Alert: {wiv_result.alert_flag}")
    print(f"  BAV: {bav_result.bav:+.4f} yrs/month | Alert: {bav_result.alert_flag}")
    print(f"  ICV: {icv_result.icv:+.4f} units/month | Alert: {icv_result.alert_flag}")
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  MCF SCORE: {mcf_result.mcf_score:.4f} [{mcf_result.band}]  ║")
    print(f"  ╚══════════════════════════════════════╝")
    print(f"\n  {mcf_result.band_message}")
    print(f"  Protocol-99 Activated: {mcf_result.protocol99_activated}")
    print(f"  Trident Firing:        {mcf_result.trident_firing}")
    print(f"  Confidence:            {mcf_result.confidence:.2f}")
    print(f"\n  Summary: {mcf_result.summary}")

    print("\n─── ICV Vascular Anomaly Score ───")
    print(f"  VAS: {icv_result.vas_score:.2f}/10 — NLR={icv_result.nlr_current:.2f} | PLR={icv_result.plr_current:.1f} | RAR={icv_result.rar_current:.4f}")
    print(f"  {icv_result.alert_reason}")

    print("\n✅ Engine test complete.")
    return mcf_result

if __name__ == "__main__":
    result = run_pipeline()
    assert result.mcf_score > 0.3, f"Expected MCF > 0.3, got {result.mcf_score}"
    print(f"\n✓ Assertion passed: MCF {result.mcf_score} > 0.3")
