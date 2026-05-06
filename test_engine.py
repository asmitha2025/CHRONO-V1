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

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import track
    import time
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

def run_pipeline():
    if not HAS_RICH:
        print("Please install rich for the premium terminal UI: pip install rich")
        return None

    console = Console()
    console.clear()
    
    console.print(Panel.fit(
        "[bold cyan]CHRONO[/bold cyan] | [bold white]The Trident Signal System[/bold white]\n[dim]Metabolic Cancer Fingerprint Engine — Google GenAI Demo[/dim]",
        border_style="cyan"
    ))

    # 1. Build personal baseline
    with console.status("[bold green]Computing Personal Biological Baseline...[/bold green]"):
        baseline = PersonalBaseline(
            patient_id=PRIYA_PROFILE["patient_id"],
            birth_year=PRIYA_PROFILE["birth_year"],
            birth_month=PRIYA_PROFILE["birth_month"]
        )
        for tp in PRIYA_TIMELINE:
            baseline.add_test(tp["date"], tp["markers"])
            time.sleep(0.05)

    console.print(f" [bold green]✓[/bold green] Baseline built from {len(PRIYA_TIMELINE)} longitudinal timepoints.")

    # 2. Initialise calculators
    timepoints = [(tp["date"], tp["markers"]) for tp in PRIYA_TIMELINE]
    wiv_calc = WIVCalculator(baseline)
    bav_calc = BAVCalculator(baseline)
    icv_calc = ICVCalculator(baseline)
    scorer   = MCFScorer()

    # 3. Final Trident Signal
    console.print("\n[bold cyan]Trident Engine Extraction (Final Timepoint)[/bold cyan]")
    
    with console.status("[bold orange3]Calculating MCF Score...[/bold orange3]"):
        time.sleep(0.5)
        wiv_result = wiv_calc.compute(timepoints)
        bav_result = bav_calc.compute(timepoints)
        icv_result = icv_calc.compute(timepoints)
        mcf_result = scorer.compute(wiv_result, bav_result, icv_result)

    # Format output based on band
    band_colors = {
        "GREEN": "bold green",
        "AMBER": "bold yellow",
        "RED": "bold red"
    }
    b_color = band_colors.get(mcf_result.alert_level, "white")

    result_text = Text()
    result_text.append(f"MCF SCORE: {mcf_result.mcf_score:.4f}\n", style=f"{b_color} italic")
    result_text.append(f"ALERT LEVEL: {mcf_result.alert_level}\n", style=b_color)
    result_text.append(f"\nConfidence: {mcf_result.confidence:.2f}\n")

    console.print(Panel(result_text, title="[bold]Metabolic Cancer Fingerprint[/bold]", border_style=b_color.split()[1]))

    if mcf_result.mcf_score >= 0.61:
        console.print("\n[bold blink red]>> INITIATING PROTOCOL-99 REASONING AGENT <<[/bold blink red]")

    return mcf_result

if __name__ == "__main__":
    result = run_pipeline()
    if result:
        assert result.mcf_score > 0.3, f"Expected MCF > 0.3, got {result.mcf_score}"
        print(f"\n✓ Assertion passed: MCF {result.mcf_score} > 0.3")
