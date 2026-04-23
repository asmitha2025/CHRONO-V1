"""
Protocol-99 Tool: validate_trident_signal
Validates if the detected metabolic signal is significant or likely noise.
"""

def validate_trident_signal(wiv_z: float, bav_z: float, icv_z: float, confidence: float) -> dict:
    """
    Validates the trident signal based on z-scores and extraction confidence.
    """
    avg_z = (abs(wiv_z) + abs(bav_z) + abs(icv_z)) / 3.0
    
    is_valid = True
    reason = "Signal validated: strong co-movement detected across markers."
    
    if confidence < 0.7:
        is_valid = False
        reason = "Signal rejected: low extraction confidence from lab reports."
    elif avg_z < 1.0:
        is_valid = False
        reason = "Signal rejected: mean deviation below noise threshold."
        
    return {
        "is_valid": is_valid,
        "reason": reason,
        "average_z_score": round(avg_z, 3),
        "confidence": confidence
    }
