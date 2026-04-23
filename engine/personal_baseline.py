"""
CHRONO — Personal Baseline Engine
Computes each user's personal biological setpoint envelope (mean ± 1.5 SD).

Scientific basis: Foy BH et al. (2025). Personalized reference intervals
for blood-test results. Nature, 637, 430–438.
"""
import json, math, os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MarkerStats:
    marker_name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min_envelope: float = 0.0
    max_envelope: float = 0.0
    is_calibrated: bool = False  # True when >= 3 data points

    def add_value(self, value: float, timestamp: str):
        self.values.append(value)
        self.timestamps.append(timestamp)
        self._recompute()

    def _recompute(self):
        n = len(self.values)
        if n == 0:
            return
        self.mean = sum(self.values) / n
        if n >= 2:
            variance = sum((v - self.mean) ** 2 for v in self.values) / (n - 1)
            self.std = math.sqrt(variance)
        else:
            self.std = 0.0
        self.min_envelope = self.mean - 1.5 * self.std
        self.max_envelope = self.mean + 1.5 * self.std
        self.is_calibrated = n >= 3

    def get_deviation(self, value: float) -> float:
        if not self.is_calibrated or self.std == 0:
            return 0.0
        return (value - self.mean) / self.std

    def is_outside_envelope(self, value: float) -> bool:
        if not self.is_calibrated:
            return False
        return value < self.min_envelope or value > self.max_envelope

    def get_percent_deviation(self, value: float) -> float:
        if self.mean == 0:
            return 0.0
        return ((value - self.mean) / self.mean) * 100.0

    def to_dict(self) -> dict:
        return {
            "marker_name": self.marker_name,
            "values": self.values,
            "timestamps": self.timestamps,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min_envelope": round(self.min_envelope, 4),
            "max_envelope": round(self.max_envelope, 4),
            "is_calibrated": self.is_calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarkerStats":
        s = cls(marker_name=data["marker_name"])
        s.values = data.get("values", [])
        s.timestamps = data.get("timestamps", [])
        s.mean = data.get("mean", 0.0)
        s.std = data.get("std", 0.0)
        s.min_envelope = data.get("min_envelope", 0.0)
        s.max_envelope = data.get("max_envelope", 0.0)
        s.is_calibrated = data.get("is_calibrated", False)
        return s


class PersonalBaseline:
    """
    Manages a patient's complete biological baseline.

    Usage:
        baseline = PersonalBaseline("patient_001", chronological_age=38)
        baseline.add_test("2018-01-10", {"ldh": 165, "albumin": 4.3, ...})
        baseline.add_test("2019-01-15", {"ldh": 170, "albumin": 4.2, ...})
        baseline.add_test("2020-01-20", {"ldh": 168, "albumin": 4.1, ...})
        norm = baseline.normalize_value("ldh", 210)  # → z-score vs personal mean
    """
    _population_norms = None

    def __init__(self, patient_id: str, chronological_age: float = 0.0):
        self.patient_id = patient_id
        self.chronological_age = chronological_age
        self.markers: Dict[str, MarkerStats] = {}
        self.test_dates: List[str] = []

    @classmethod
    def _load_norms(cls) -> dict:
        if cls._population_norms is None:
            p = os.path.join(os.path.dirname(__file__), "population_norms.json")
            with open(p) as f:
                cls._population_norms = json.load(f)
        return cls._population_norms

    def add_test(self, test_date: str, marker_values: Dict[str, float]):
        self.test_dates.append(test_date)
        for name, val in marker_values.items():
            if name not in self.markers:
                self.markers[name] = MarkerStats(marker_name=name)
            self.markers[name].add_value(val, test_date)
        self._compute_derived_ratios(test_date, marker_values)

    def _compute_derived_ratios(self, date: str, mv: Dict[str, float]):
        if "neutrophils" in mv and "lymphocytes" in mv and mv["lymphocytes"] > 0:
            nlr = mv["neutrophils"] / mv["lymphocytes"]
            if "nlr" not in self.markers:
                self.markers["nlr"] = MarkerStats(marker_name="nlr")
            self.markers["nlr"].add_value(round(nlr, 3), date)
        if "platelets" in mv and "lymphocytes" in mv and mv["lymphocytes"] > 0:
            plr = mv["platelets"] / mv["lymphocytes"]
            if "plr" not in self.markers:
                self.markers["plr"] = MarkerStats(marker_name="plr")
            self.markers["plr"].add_value(round(plr, 2), date)
        if "rdw" in mv and "albumin" in mv and mv["albumin"] > 0:
            rar = mv["rdw"] / mv["albumin"]
            if "rar" not in self.markers:
                self.markers["rar"] = MarkerStats(marker_name="rar")
            self.markers["rar"].add_value(round(rar, 4), date)

    def normalize_value(self, marker_name: str, value: float) -> float:
        """Return (value - personal_mean) / personal_std. Falls back to population norms."""
        if marker_name in self.markers and self.markers[marker_name].is_calibrated:
            s = self.markers[marker_name]
            if s.std > 0:
                return (value - s.mean) / s.std
        norms = self._load_norms()
        info = norms.get("markers", {}).get(marker_name, {})
        pop_mean = info.get("population_mean", 0)
        pop_std = info.get("population_std", 1)
        return (value - pop_mean) / pop_std if pop_std > 0 else 0.0

    def get_marker_history(self, marker_name: str) -> List[Tuple[str, float]]:
        if marker_name not in self.markers:
            return []
        s = self.markers[marker_name]
        return list(zip(s.timestamps, s.values))

    def evaluate_test(self, marker_values: Dict[str, float]) -> Dict[str, dict]:
        norms = self._load_norms()
        results = {}
        for name, value in marker_values.items():
            r = {"current_value": value, "personal_mean": 0.0, "personal_std": 0.0,
                 "z_score": 0.0, "pct_deviation": 0.0, "flagged": False,
                 "population_status": "UNKNOWN", "personal_status": "UNCALIBRATED"}
            info = norms.get("markers", {}).get(name, {})
            if info:
                lo, hi = info.get("reference_low", 0), info.get("reference_high", 1e9)
                r["population_status"] = "NORMAL" if lo <= value <= hi else "ABNORMAL"
            if name in self.markers:
                s = self.markers[name]
                r["personal_mean"] = s.mean
                r["personal_std"] = s.std
                if s.is_calibrated:
                    r["z_score"] = round(s.get_deviation(value), 3)
                    r["pct_deviation"] = round(s.get_percent_deviation(value), 2)
                    r["flagged"] = s.is_outside_envelope(value)
                    r["personal_status"] = "FLAGGED" if r["flagged"] else "NORMAL"
            results[name] = r
        return results

    def get_calibration_status(self) -> Dict[str, bool]:
        """Get calibration status for all tracked markers."""
        return {
            name: stats.is_calibrated for name, stats in self.markers.items()
        }

    def to_dict(self) -> dict:
        return {"patient_id": self.patient_id, "chronological_age": self.chronological_age,
                "test_dates": self.test_dates,
                "markers": {n: s.to_dict() for n, s in self.markers.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "PersonalBaseline":
        b = cls(patient_id=data["patient_id"], chronological_age=data.get("chronological_age", 0.0))
        b.test_dates = data.get("test_dates", [])
        b.markers = {n: MarkerStats.from_dict(sd) for n, sd in data.get("markers", {}).items()}
        return b

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PersonalBaseline":
        with open(filepath) as f:
            return cls.from_dict(json.load(f))
