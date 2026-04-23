"""
Protocol-99 Tool: compute_vascular_anomaly_score
Performs a deep-dive analysis of the Immune Collapse Velocity (ICV) components.
"""

def compute_vascular_anomaly_score(nlr: float, plr: float, rar: float) -> dict:
    """
    Computes the Vascular Anomaly Score (VAS) 0-10.
    """
    # Simple logic for the demo, matching the ICVCalculator logic
    # In a real system, this would involve more complex pattern matching
    score = (nlr * 0.4 + plr/50 * 0.3 + rar*2 * 0.3)
    score = min(10.0, score)
    
    classification = "NORMAL"
    if score > 7.0:
        classification = "CRITICAL"
    elif score > 4.0:
        classification = "WATCH"
        
    return {
        "vas_score": round(score, 2),
        "classification": classification,
        "components": {
            "nlr": nlr,
            "plr": plr,
            "rar": rar
        }
    }
