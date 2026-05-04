"""Protocol-99 Agent Tools — five specialised tools for clinical triage."""

from .validate_signal import validate_trident_signal
from .query_history import query_personal_history
from .vascular_anomaly import compute_vascular_anomaly_score
from .generate_dossier import generate_triage_dossier
from .escalate_oncologist import escalate_to_oncologist

__all__ = [
    "validate_trident_signal",
    "query_personal_history",
    "compute_vascular_anomaly_score",
    "generate_triage_dossier",
    "escalate_to_oncologist"
]
