"""
CHRONO — Unsloth Fine-Tuning Data Preparation
==============================================
Converts raw extraction data into the instruction-following format 
required for fine-tuning Gemma 4 E4B.

Target Format: Alpaca / ShareGPT instruction pairs.
Purpose: Improve Gemma 4's accuracy in extracting structured JSON from 
heterogeneous, low-quality Indian lab report photographs.
"""

import json
import os
from typing import List, Dict

def create_alpaca_pair(image_description: str, ground_truth_json: dict) -> dict:
    """Creates a single instruction-following pair."""
    return {
        "instruction": "Extract all blood test values from this lab report image and return as structured JSON.",
        "input": image_description, # In real fine-tuning, this would be the image token
        "output": json.dumps(ground_truth_json, indent=2)
    }

def generate_training_data(output_path: str):
    """
    Generates a synthetic training set with 20 examples for the fine-tuning demo.
    """
    dataset = []
    
    scenarios = [
        ("Apollo Diagnostics, Chennai", "Jan 2024", [("LDH", 214, "U/L"), ("RDW", 14.2, "%")]),
        ("Dr. Lal PathLabs", "Nov 2023", [("Hemoglobin", 12.1, "g/dL"), ("WBC", 10.5, "10^3/uL")]),
        ("Metropolis Healthcare", "Feb 2024", [("Glucose", 110, "mg/dL"), ("HbA1c", 6.2, "%")]),
        ("SRL Diagnostics", "Dec 2023", [("Creatinine", 1.1, "mg/dL"), ("BUN", 18, "mg/dL")]),
        ("Thyrocare", "Oct 2023", [("TSH", 4.2, "uIU/mL"), ("T4", 1.2, "ng/dL")]),
        ("Medall Healthcare", "Mar 2024", [("Albumin", 3.8, "g/dL"), ("CRP", 4.5, "mg/L")]),
        ("AIG Hospitals", "Jan 2024", [("ALP", 102, "U/L"), ("ALT", 35, "U/L")]),
        ("Max Healthcare", "Nov 2023", [("Platelets", 210, "10^3/uL"), ("MCV", 88, "fL")]),
        ("Fortis Memorial", "Sep 2023", [("Bilirubin", 0.9, "mg/dL"), ("AST", 28, "U/L")]),
        ("Manipal Hospitals", "Aug 2023", [("Calcium", 9.4, "mg/dL"), ("Potassium", 4.1, "mmol/L")]),
        ("Aster DM", "Jul 2023", [("Sodium", 140, "mmol/L"), ("Chloride", 102, "mmol/L")]),
        ("KIMS Hospitals", "Jun 2023", [("Magnesium", 2.0, "mg/dL"), ("Phosphorus", 3.4, "mg/dL")]),
        ("Narayana Health", "May 2023", [("Iron", 85, "ug/dL"), ("Ferritin", 120, "ng/mL")]),
        ("Global Hospitals", "Apr 2023", [("Vitamin D", 22, "ng/mL"), ("Vitamin B12", 340, "pg/mL")]),
        ("Care Hospitals", "Mar 2023", [("Uric Acid", 5.8, "mg/dL"), ("Triglycerides", 160, "mg/dL")]),
        ("Cloudnine", "Feb 2023", [("HDL", 45, "mg/dL"), ("LDL", 130, "mg/dL")]),
        ("Rainbow Children's", "Jan 2023", [("Total Cholesterol", 195, "mg/dL"), ("VLDL", 32, "mg/dL")]),
        ("Vijaya Diagnostic", "Dec 2022", [("ESR", 15, "mm/hr"), ("INR", 1.0, "ratio")]),
        ("Neuberg Diagnostics", "Nov 2022", [("PSA", 2.1, "ng/mL"), ("CEA", 1.5, "ng/mL")]),
        ("Hitech Diagnostic", "Oct 2022", [("Amylase", 65, "U/L"), ("Lipase", 40, "U/L")]),
    ]

    for lab, date, markers in scenarios:
        dataset.append(create_alpaca_pair(
            f"A lab report from {lab} dated {date}.",
            {
                "test_date": date,
                "lab_name": lab,
                "markers": [{"name": n, "value": v, "unit": u} for n, v, u in markers]
            }
        ))

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"[CHRONO] Training data prepared: {len(dataset)} examples -> {output_path}")

if __name__ == "__main__":
    os.makedirs("training/data", exist_ok=True)
    generate_training_data("training/data/gemma4_finetune_data.json")
