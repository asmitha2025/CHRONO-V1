"""
Protocol-99 Tool: query_personal_history
Retrieves longitudinal history for a specific marker.
"""
from typing import List, Tuple
from engine.personal_baseline import PersonalBaseline

def query_personal_history(baseline: PersonalBaseline, marker_name: str) -> dict:
    """
    Queries the historical values and timestamps for a marker.
    """
    history = baseline.get_marker_history(marker_name)
    if not history:
        return {"status": "error", "message": f"No history found for marker: {marker_name}"}
    
    values = [v for _, v in history]
    dates = [d for d, _ in history]
    
    return {
        "status": "success",
        "marker_name": marker_name,
        "history_points": len(history),
        "latest_value": values[-1],
        "first_value": values[0],
        "trend": "increasing" if values[-1] > values[0] else "decreasing",
        "data": history
    }
