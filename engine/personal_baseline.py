from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

@dataclass
class PersonalBaseline:
    patient_id: str
    birth_year: int
    birth_month: int = 6
    timepoints: List = field(default_factory=list)
    _stats_cache: Dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_timepoints(cls, patient_id, birth_year, timepoints, birth_month=6):
        sorted_tp = sorted(timepoints, key=lambda t: t[0])
        inst = cls(patient_id, birth_year, birth_month, sorted_tp)
        inst._build_stats_cache()
        return inst

    def _build_stats_cache(self):
        """Builds mean/std cache for all markers."""
        markers = {}
        for _, m in self.timepoints:
            for k, v in m.items():
                if k not in markers: markers[k] = []
                markers[k].append(v)
        
        for k, vals in markers.items():
            if len(vals) >= 2:
                self._stats_cache[k] = {
                    "mean": statistics.mean(vals),
                    "stdev": statistics.pstdev(vals),
                    "sorted": sorted(vals),
                    "n": len(vals)
                }

    def zscore(self, marker: str, value: float) -> Optional[float]:
        stats = self._stats_cache.get(marker)
        if stats is None or stats["stdev"] == 0: return None
        return (value - stats["mean"]) / stats["stdev"]

    def percentile_rank(self, marker: str, value: float) -> Optional[float]:
        stats = self._stats_cache.get(marker)
        if stats is None: return None
        vals = stats["sorted"]
        below = sum(1 for v in vals if v < value)
        equal = sum(1 for v in vals if v == value)
        return round((below + 0.5 * equal) / stats["n"] * 100.0, 1)

    def chronological_age_at(self, date_str: str) -> float:
        """Exact age in years -- used by BAVCalculator for every timepoint."""
        d = date.fromisoformat(date_str)
        birth = date(self.birth_year, self.birth_month, 15)
        return (d - birth).days / 365.25

    def add_test(self, test_date: str, marker_values: Dict[str, float]):
        # Add derived ratios
        mv = marker_values.copy()
        if "neutrophils" in mv and "lymphocytes" in mv and mv["lymphocytes"] > 0:
            mv["nlr"] = mv["neutrophils"] / mv["lymphocytes"]
        if "platelets" in mv and "lymphocytes" in mv and mv["lymphocytes"] > 0:
            mv["plr"] = mv["platelets"] / mv["lymphocytes"]
        if "rdw" in mv and "albumin" in mv and mv["albumin"] > 0:
            mv["rar"] = mv["rdw"] / mv["albumin"]
            
        self.timepoints.append((test_date, mv))
        self._build_stats_cache()
    
    def get_calibration_status(self) -> Dict[str, bool]:
        return {k: v["n"] >= 3 for k, v in self._stats_cache.items()}
    
    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "birth_year": self.birth_year,
            "birth_month": self.birth_month,
            "timepoints": self.timepoints
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PersonalBaseline":
        return cls.from_timepoints(
            data["patient_id"], data["birth_year"], 
            data["timepoints"], data.get("birth_month", 6)
        )
