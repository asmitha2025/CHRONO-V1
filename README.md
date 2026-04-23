# CHRONO | The Trident Signal System
### Gemma 4 Good Hackathon 2026 — Submission

CHRONO is an AI-driven metabolic anomaly detection system that identifies pre-cancerous signals from routine blood test history, filling a 3-to-5-year diagnostic blind spot in oncology.

## 🚀 Key Innovation
Instead of comparing blood values against population averages, CHRONO computes each person's own biological setpoint envelope (mean ± 1.5 SD). This allows it to detect subtle velocities (WIV, BAV, ICV) that indicate malignant metabolic reprogramming even while markers remain within "normal" lab ranges.

## 🛠️ Technology Stack
- **Gemma 4 Vision (E4B):** Local document OCR and structured marker extraction.
- **Gemma 4 26B:** Protocol-99 ReAct agent for clinical triage.
- **Trident Signal Engine:** Custom Python engine for WIV, BAV, and ICV calculation.
- **Web Demo:** High-fidelity "Metabolic Nocturne" dashboard (HTML/CSS/JS).

## 📁 Project Structure
- `engine/`: The core Trident Signal calculators.
- `agent/`: Protocol-99 reasoning agent and clinical tools.
- `ingestion/`: Gemma 4 vision-powered lab report parsing.
- `demo/`: Web-based interactive dashboard.
- `notebooks/`: Technical Kaggle notebook for judging.
- `data/`: Synthetic generator for demo patient "Priya".

## 🚦 Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the engine test: `python test_engine.py`
3. Launch the demo: Open `demo/index.html` in any modern browser.

## ⚖️ License
Apache 2.0 (as required by hackathon rules).

---
*Built by Asmitha M & Hariharan M for the Gemma 4 Good Hackathon.*
