"""
CHRONO — Trident Signal Engine
The core metabolic cancer fingerprint computation engine.

Components:
    - WIVCalculator: Warburg Index Velocity (aerobic glycolysis shift rate)
    - BAVCalculator: Biological Age Velocity (PhenoAge acceleration rate)
    - ICVCalculator: Immune Collapse Velocity (immune ratio deterioration rate)
    - MCFScorer: Metabolic Cancer Fingerprint aggregator
    - PersonalBaseline: Personal setpoint envelope computation
"""

from .personal_baseline import PersonalBaseline
from .wiv_calculator import WIVCalculator
from .bav_calculator import BAVCalculator
from .icv_calculator import ICVCalculator
from .mcf_scorer import MCFScorer

__all__ = [
    "PersonalBaseline",
    "WIVCalculator",
    "BAVCalculator",
    "ICVCalculator",
    "MCFScorer",
]
