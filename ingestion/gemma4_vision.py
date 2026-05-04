"""
CHRONO — Gemma 4 Vision Ingestion Layer
========================================
Extracts structured blood test data from any lab report image/PDF
using Gemma 4 E4B's native multimodal vision capability.

Supports three backends:
  1. GOOGLE_AI  — Google AI Studio API (recommended for demo, fast)
  2. OLLAMA     — Local Ollama server (privacy-first, offline)
  3. HUGGINGFACE — HuggingFace transformers (Kaggle notebook GPU)

Set via environment variable: GEMMA_BACKEND=google_ai|ollama|huggingface

Model used: gemma-4-e4b (4.5B effective params, native multimodal)
  - On-device capable via LiteRT
  - 128K context window
  - Supports image + text input natively

Scientific context:
  Gemma 4 E4B makes CHRONO architecturally possible for the first time —
  no prior open model could reliably extract structured medical data from
  heterogeneous lab report formats at consumer scale (April 2026).
"""

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Backend detection ────────────────────────────────────────────
BACKEND = os.getenv("GEMMA_BACKEND", "google_ai").lower()

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "google/gemma-4-e4b-it")

# ── Gemma 4 extraction prompt ────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a medical document parser for CHRONO, an AI health system.
Extract ALL blood test values from this lab report image or audio dictation.
Return ONLY valid JSON with this exact schema — no other text:

{
  "test_date": "YYYY-MM-DD",
  "lab_name": "string or null",
  "patient_name": "string or null",
  "markers": [
    {
      "name": "string",
      "value": number,
      "unit": "string",
      "reference_low": number or null,
      "reference_high": number or null,
      "flag": "H" or "L" or "N" or null,
      "confidence": number between 0.0 and 1.0
    }
  ]
}

Rules:
- Include ALL markers visible in the report
- Use the exact value printed, do not round
- confidence: 1.0 = clearly readable, 0.5 = partially obscured, 0.0 = guessed
- flag: H=High, L=Low, N=Normal, null=not printed
- If test_date not visible, use null
- Do NOT add any text outside the JSON object"""


@dataclass
class ExtractedMarker:
    name: str
    value: float
    unit: str
    reference_low: Optional[float]
    reference_high: Optional[float]
    flag: Optional[str]
    confidence: float
    normalized_name: Optional[str] = None  # set by marker synonym lookup

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "value": self.value,
            "unit": self.unit,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
            "flag": self.flag,
            "confidence": self.confidence,
        }


@dataclass
class LabReport:
    test_date: Optional[str]
    lab_name: Optional[str]
    patient_name: Optional[str]
    markers: List[ExtractedMarker] = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)
    extraction_backend: str = ""
    extraction_success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "test_date": self.test_date,
            "lab_name": self.lab_name,
            "patient_name": self.patient_name,
            "markers": [m.to_dict() for m in self.markers],
            "extraction_backend": self.extraction_backend,
            "extraction_success": self.extraction_success,
            "error_message": self.error_message,
        }

    def get_marker_dict(self, min_confidence: float = 0.75) -> Dict[str, float]:
        """Return {normalized_name: value} for markers above confidence threshold."""
        result = {}
        for m in self.markers:
            name = m.normalized_name or m.name.lower().replace(" ", "_")
            if m.confidence >= min_confidence:
                result[name] = m.value
        return result


def _load_synonyms() -> Dict[str, str]:
    """Load marker synonym mapping from JSON file."""
    path = Path(__file__).parent / "marker_synonyms.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _normalize_marker_name(name: str, synonyms: Dict[str, str]) -> str:
    """Map lab report marker names to CHRONO canonical names."""
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    return synonyms.get(cleaned, cleaned)


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_gemma_response(response_text: str) -> dict:
    """Extract JSON from Gemma 4 response, handling markdown code blocks."""
    text = response_text.strip()
    # Remove markdown code fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?\n?", "", text).strip()
    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"No valid JSON found in Gemma response: {text[:200]}")


# ── Backend implementations ───────────────────────────────────────

def _extract_via_google_ai(image_path: str) -> dict:
    """Extract using Google AI Studio API (Gemma 4 E4B)."""
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    ext = Path(image_path).suffix.lower().lstrip(".")
    is_audio = ext in ["mp3", "wav", "m4a", "ogg", "flac"]
    
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    if is_audio:
        mime = {"mp3": "audio/mp3", "wav": "audio/wav", "m4a": "audio/m4a", "ogg": "audio/ogg", "flac": "audio/flac"}.get(ext, "audio/mp3")
        media_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
        print(f"[CHRONO Ingestion] Sending lab report audio to Gemma 4...")
    else:
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        media_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)
        print(f"[CHRONO Ingestion] Sending lab report image to Gemma 4 Vision...")

    response = client.models.generate_content(
        model='gemma-4-e4b-it',
        contents=[EXTRACTION_SYSTEM_PROMPT, media_part],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
        )
    )
    return _parse_gemma_response(response.text)


def _extract_via_ollama(image_path: str) -> dict:
    """Extract using local Ollama with Gemma 4 E4B."""
    import requests
    img_b64 = _image_to_base64(image_path)
    payload = {
        "model": "gemma4:e4b",
        "prompt": EXTRACTION_SYSTEM_PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return _parse_gemma_response(resp.json()["response"])


def _extract_via_huggingface(image_path: str) -> dict:
    """Extract using HuggingFace transformers (for Kaggle GPU notebook)."""
    from PIL import Image
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = HUGGINGFACE_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=EXTRACTION_SYSTEM_PROMPT, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return _parse_gemma_response(text)


def _mock_extract(image_path: str) -> dict:
    """
    Mock extraction for testing without a live Gemma 4 endpoint.
    Returns a realistic blood test result matching the Priya demo scenario.
    """
    print(f"[CHRONO] MOCK extraction active (set GEMMA_BACKEND to use real model)")
    return {
        "test_date": "2024-01-18",
        "lab_name": "Apollo Diagnostics, Chennai",
        "patient_name": "Priya",
        "markers": [
            {"name": "LDH", "value": 214.0, "unit": "U/L",
             "reference_low": 140.0, "reference_high": 280.0, "flag": "N", "confidence": 0.97},
            {"name": "RDW", "value": 14.2, "unit": "%",
             "reference_low": 11.5, "reference_high": 14.5, "flag": "N", "confidence": 0.96},
            {"name": "Glucose (Fasting)", "value": 101.0, "unit": "mg/dL",
             "reference_low": 70.0, "reference_high": 100.0, "flag": "H", "confidence": 0.98},
            {"name": "Albumin", "value": 3.95, "unit": "g/dL",
             "reference_low": 3.5, "reference_high": 5.5, "flag": "N", "confidence": 0.97},
            {"name": "Creatinine", "value": 0.89, "unit": "mg/dL",
             "reference_low": 0.6, "reference_high": 1.2, "flag": "N", "confidence": 0.99},
            {"name": "C-Reactive Protein", "value": 3.8, "unit": "mg/L",
             "reference_low": 0.0, "reference_high": 3.0, "flag": "H", "confidence": 0.95},
            {"name": "ALP", "value": 81.0, "unit": "U/L",
             "reference_low": 44.0, "reference_high": 147.0, "flag": "N", "confidence": 0.96},
            {"name": "MCV", "value": 91.0, "unit": "fL",
             "reference_low": 80.0, "reference_high": 100.0, "flag": "N", "confidence": 0.97},
            {"name": "WBC", "value": 7.8, "unit": "×10³/µL",
             "reference_low": 4.5, "reference_high": 11.0, "flag": "N", "confidence": 0.98},
            {"name": "Lymphocyte %", "value": 26.5, "unit": "%",
             "reference_low": 20.0, "reference_high": 40.0, "flag": "N", "confidence": 0.96},
            {"name": "Neutrophils", "value": 5.4, "unit": "×10³/µL",
             "reference_low": 1.8, "reference_high": 7.7, "flag": "N", "confidence": 0.97},
            {"name": "Lymphocytes", "value": 1.73, "unit": "×10³/µL",
             "reference_low": 1.0, "reference_high": 4.8, "flag": "N", "confidence": 0.97},
            {"name": "Platelets", "value": 290.0, "unit": "×10³/µL",
             "reference_low": 150.0, "reference_high": 400.0, "flag": "N", "confidence": 0.98},
            {"name": "Haemoglobin", "value": 13.5, "unit": "g/dL",
             "reference_low": 12.0, "reference_high": 17.5, "flag": "N", "confidence": 0.99},
        ],
    }


# ── Public API ────────────────────────────────────────────────────

class Gemma4VisionExtractor:
    """
    Gemma 4 E4B vision-powered lab report extractor.

    Reads any blood test document from a photograph and returns
    structured, normalised marker data. Patient data never leaves
    the device when using OLLAMA or HUGGINGFACE backends.

    Usage:
        extractor = Gemma4VisionExtractor()
        report = extractor.extract("path/to/blood_test.jpg")
        markers = report.get_marker_dict()
        print(markers)  # {"ldh": 214.0, "albumin": 3.95, ...}
    """

    def __init__(self):
        self.synonyms = _load_synonyms()
        self.backend = BACKEND
        print(f"[CHRONO] Gemma 4 Vision extractor initialized — backend: {self.backend}")

    def extract(self, image_path: str) -> LabReport:
        """
        Extract blood test data from an image file.

        Args:
            image_path: Path to lab report image (JPEG, PNG, WEBP)

        Returns:
            LabReport with extracted and normalised marker data
        """
        try:
            if self.backend == "google_ai" and GOOGLE_AI_API_KEY:
                raw = _extract_via_google_ai(image_path)
            elif self.backend == "ollama":
                raw = _extract_via_ollama(image_path)
            elif self.backend == "huggingface":
                raw = _extract_via_huggingface(image_path)
            else:
                raw = _mock_extract(image_path)

            markers = []
            for m in raw.get("markers", []):
                try:
                    marker = ExtractedMarker(
                        name=m["name"],
                        value=float(m["value"]),
                        unit=m.get("unit", ""),
                        reference_low=m.get("reference_low"),
                        reference_high=m.get("reference_high"),
                        flag=m.get("flag"),
                        confidence=float(m.get("confidence", 0.8)),
                    )
                    marker.normalized_name = _normalize_marker_name(marker.name, self.synonyms)
                    markers.append(marker)
                except (KeyError, ValueError, TypeError):
                    continue

            return LabReport(
                test_date=raw.get("test_date"),
                lab_name=raw.get("lab_name"),
                patient_name=raw.get("patient_name"),
                markers=markers,
                raw_json=raw,
                extraction_backend=self.backend,
                extraction_success=True,
            )

        except Exception as e:
            return LabReport(
                test_date=None, lab_name=None, patient_name=None,
                markers=[], raw_json={},
                extraction_backend=self.backend,
                extraction_success=False,
                error_message=str(e),
            )

    def extract_and_normalise(self, image_path: str, min_confidence: float = 0.75) -> Dict[str, float]:
        """Convenience method — extract and return normalised marker dict directly."""
        report = self.extract(image_path)
        return report.get_marker_dict(min_confidence)
