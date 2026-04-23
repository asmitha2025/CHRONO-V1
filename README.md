# CHRONO | The Trident Signal System
### World's First Metabolic Cancer Fingerprint Engine
**Gemma 4 Good Hackathon 2026 — Health & Sciences Track**

---

## 🌌 Overview
**CHRONO** is a revolutionary AI-driven metabolic anomaly detection system designed to identify pre-cancerous signals from routine blood test history. By computing biological velocity against a personal baseline, CHRONO fills a critical **3-to-5-year diagnostic blind spot** in oncology—detecting the metabolic "fingerprint" of cancer years before structural changes are visible on imaging (PET/CT/MRI).

> "Biology precedes structure. CHRONO listens to the biology before the structure breaks."

---

## 🔱 The Trident Signal™
The core innovation of CHRONO is the **Trident Signal**, a simultaneous analysis of three independent metabolic velocities that co-move during malignant reprogramming.

| Signal | What It Measures | Scientific Basis | Key Markers |
| :--- | :--- | :--- | :--- |
| **Warburg Index Velocity (WIV)** | Rate of metabolic shift toward aerobic glycolysis (the "Warburg Effect"). | Warburg 1924, Nobel 1931 | LDH, Glucose, RDW trend |
| **Biological Age Velocity (BAV)** | Rate at which biological age diverges from chronological age. | UK Biobank (n=308,156) | Albumin, CRP, ALP, MCV, RDW |
| **Immune Collapse Velocity (ICV)** | Rate of deterioration in immune ratio balance (Neutrophil-to-Lymphocyte Ratio). | Sci Rep 2025, NLR meta-analysis | NLR, PLR, RAR, PNI |

---

## 🧪 Scientific Methodology: Personal Baseline Calibration
Standard medicine compares your blood to a **population average** (the 2.5th–97.5th percentile of strangers). CHRONO ignores the population and computes your **Personal Biological Setpoint Envelope** (Mean ± 1.5 SD).

*   **The Normal Range Trap:** A Hemoglobin of 13.1 is "Normal" for the population, but if your personal mean is 15.2, a drop to 13.1 represents a **14% biological collapse**. CHRONO flags this velocity; a standard lab report discards it.
*   **The Trident Rule:** A single signal might be noise (stress, infection). The Trident Signal only fires when **all three independent processes** co-move, providing a high-confidence indicator of systemic metabolic stress.

---

## 🛠️ Technology Stack & Architecture
CHRONO is built on a multi-layer on-device and cloud-enclave architecture.

### 1. Document Ingestion (Gemma 4 Vision)
*   **Model:** Gemma 4 E4B (on-device via LiteRT).
*   **Function:** Multimodal OCR that reads any lab report (PDF, photo, printout) and extracts structured JSON with unit normalization.

### 2. Trident Signal Engine (Fine-Tuned Gemma 4 E4B)
*   **Model:** Gemma 4 E4B fine-tuned using **Unsloth** on MIMIC-IV and PubMedQA.
*   **Function:** Computes WIV, BAV, and ICV scores locally on the user's phone, ensuring maximum privacy.

### 3. Protocol-99 Agentic Triage (Gemma 4 26B)
*   **Model:** Gemma 4 26B ReAct Agent in a Private Cloud Enclave.
*   **Function:** Activated when the **Metabolic Cancer Fingerprint (MCF)** score crosses 0.61. The agent uses tools to:
    *   `validate_signal`: Eliminate confounders (medication, dehydration).
    *   `query_history`: Cross-reference against 10-year trends.
    *   `generate_dossier`: Create a structured, oncology-grade triage report.
    *   `escalate_oncologist`: Securely route the dossier for human review.

---

## 📁 Project Structure
```bash
├── agent/            # Protocol-99 ReAct agent and clinical triage tools
├── api/              # Secure FastAPI endpoints for agentic reasoning
├── data/             # Synthetic patient generator (NHANES-calibrated)
├── demo/             # "Metabolic Nocturne" High-fidelity dashboard
├── docs/             # Technical writeups and scientific references
├── engine/           # The Core Trident Signal (WIV, BAV, ICV) calculators
├── ingestion/        # Gemma 4 Vision pipeline for lab report parsing
├── notebooks/        # Technical Kaggle notebook for judging
├── training/         # Unsloth fine-tuning scripts and data prep
└── requirements.txt  # Production dependencies
```

---

## 🚦 Getting Started

### Prerequisites
*   Python 3.10+
*   Google AI Studio API Key (for Gemma 4 backend)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/asmitha2025/CHRONO-V1.git
    cd CHRONO-V1
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Environment:**
    Create a `.env` file and add your `GOOGLE_AI_API_KEY`.

### Running the System
*   **Test the Signal Engine:** `python test_engine.py`
*   **Launch the Dashboard:** Open `demo/index.html` in your browser to view the high-fidelity patient monitoring interface.

---

## ⚖️ License
Distributed under the **Apache 2.0 License**. See `LICENSE` for more information.

---

## ✨ Team - The CHRONO Collective
*   **Asmitha M** - AI Architecture & Scientific Lead
*   **Hariharan M** - Systems Engineering & UI/UX

*Built for the Gemma 4 Good Hackathon 2026.*
