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
    Generates a synthetic training set for the fine-tuning demo.
    In a real scenario, this would load real anonymised lab reports.
    """
    dataset = []
    
    # Example 1: Standard Apollo Report
    dataset.append(create_alpaca_pair(
        "A photograph of a lab report from Apollo Diagnostics dated Jan 2024.",
        {
            "test_date": "2024-01-18",
            "markers": [
                {"name": "LDH", "value": 214, "unit": "U/L"},
                {"name": "RDW", "value": 14.2, "unit": "%"}
            ]
        }
    ))
    
    # Example 2: Handwritten/Noisy Report
    dataset.append(create_alpaca_pair(
        "A low-light image of a CBC report with handwritten notes in the margins.",
        {
            "test_date": "2023-11-05",
            "markers": [
                {"name": "Hemoglobin", "value": 12.1, "unit": "g/dL"},
                {"name": "WBC", "value": 10.5, "unit": "10^3/uL"}
            ]
        }
    ))

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"[CHRONO] Training data prepared: {len(dataset)} examples -> {output_path}")

if __name__ == "__main__":
    os.makedirs("training/data", exist_ok=True)
    generate_training_data("training/data/gemma4_finetune_data.json")
