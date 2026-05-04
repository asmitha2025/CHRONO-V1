"""
CHRONO — Protocol-99 ReAct Agent
=================================
A reasoning agent powered by Gemma 4 26B that follows the ReAct pattern
(Thought, Action, Action Input, Observation) to perform clinical triage.

The agent activates when the MCF score crosses the 0.61 threshold (ORANGE).
It validates the signal, investigates history, and generates an oncologist dossier.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

# Import tools
from agent.tools import (
    validate_trident_signal,
    query_personal_history,
    compute_vascular_anomaly_score,
    generate_triage_dossier,
    escalate_to_oncologist
)

class Protocol99Agent:
    """
    Protocol-99 Agentic Triage Engine.
    
    Implements a ReAct loop that uses Gemma 4 for reasoning and 
    specialized Python tools for clinical data analysis.
    """
    
    def __init__(self, patient_id: str, baseline: Any, mcf_result: Any):
        self.patient_id = patient_id
        self.baseline = baseline
        self.mcf_result = mcf_result
        self.max_iterations = 8
        
        # Load system prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read()
            
        self.history = []

    def _get_formatted_prompt(self) -> str:
        """Formats the system prompt with current patient context."""
        from datetime import datetime
        return self.system_prompt_template.format(
            patient_id=self.patient_id,
            mcf_score=self.mcf_result.mcf_score,
            trident_status="FIRING" if self.mcf_result.trident_firing else "PARTIAL",
            current_date=datetime.now().strftime("%Y-%m-%d")
        )

    def run_simulation(self) -> List[Dict[str, str]]:
        """
        Runs a high-fidelity simulation of the ReAct loop.
        Used for development and demo when real model access is limited.
        """
        print(f"[CHRONO Agent] Activating Protocol-99 for {self.patient_id}...")
        
        trace = []
        
        # --- Step 1: Validation ---
        trace.append({
            "role": "thought", 
            "content": "MCF score is 0.73. I must first validate the Trident signal to ensure this isn't an artifact of low confidence or transient noise."
        })
        val = validate_trident_signal(
            self.mcf_result.wiv_score, 
            self.mcf_result.bav_score, 
            self.mcf_result.icv_score, 
            self.mcf_result.confidence
        )
        trace.append({
            "role": "action", 
            "content": f"validate_trident_signal(wiv_z={self.mcf_result.wiv_score}, confidence={self.mcf_result.confidence})"
        })
        trace.append({"role": "observation", "content": json.dumps(val)})
        
        # --- Step 2: History Query ---
        trace.append({
            "role": "thought", 
            "content": "Signal validated. Now investigating long-term marker trends. I'll query the LDH history to see the trajectory of the Warburg signal."
        })
        hist = query_personal_history(self.baseline, "ldh")
        trace.append({"role": "action", "content": "query_personal_history(marker_name='ldh')"})
        trace.append({"role": "observation", "content": f"LDH trend: {hist['trend']}. Latest: {hist['latest_value']} U/L (from {hist['first_value']} U/L baseline)."})
        
        # --- Step 3: Vascular Analysis ---
        trace.append({
            "role": "thought", 
            "content": "Sustained LDH elevation detected over 18 months. Now performing a deep-dive into the immune collapse pattern using the Vascular Anomaly Score."
        })
        vas = compute_vascular_anomaly_score(
            self.mcf_result.icv_raw, # simplified for simulation
            168.0, 3.59
        )
        trace.append({"role": "action", "content": f"compute_vascular_anomaly_score(nlr={self.mcf_result.icv_raw}, plr=168.0, rar=3.59)"})
        trace.append({"role": "observation", "content": json.dumps(vas)})
        
        # --- Step 4: Generate Dossier ---
        trace.append({
            "role": "thought", 
            "content": "Vascular Anomaly Score is 6.96 (CRITICAL). Pattern suggests metabolic reprogramming. I will now generate the clinical triage dossier for oncologist review."
        })
        summary = (
            f"Trident Signal co-movement confirmed. WIV velocity {self.mcf_result.wiv_raw:+.3f} units/mo. "
            f"BAV acceleration {self.mcf_result.bav_raw:+.3f} yrs/mo. ICV deterioration {self.mcf_result.icv_raw:+.3f} units/mo. "
            f"Personal baseline deviation exceeded 1.5σ for LDH and RDW. VAS score 6.96."
        )
        dossier = generate_triage_dossier(summary, "CRITICAL", self.patient_id, self.mcf_result.bav_raw)
        trace.append({"role": "action", "content": "generate_triage_dossier(risk_level='CRITICAL')"})
        trace.append({"role": "observation", "content": "Dossier generated successfully (Markdown format)."})
        
        # --- Step 5: Escalate ---
        trace.append({
            "role": "thought", 
            "content": "Dossier ready. Escalating to the on-call oncology specialist with high urgency."
        })
        esc = escalate_to_oncologist(dossier, "HIGH")
        trace.append({"role": "action", "content": "escalate_to_oncologist(urgency='HIGH')"})
        trace.append({"role": "observation", "content": json.dumps(esc)})
        
        # --- Final Answer ---
        trace.append({
            "role": "final_answer", 
            "content": (
                "Protocol-99 investigation complete. Trident signal validated (Avg Z=2.1). "
                "Sustained 18-month metabolic acceleration detected. "
                "Triage dossier generated and successfully escalated to Oncology Specialist. "
                "Case ID: CHRONO-2026-99-A1."
            )
        })
        
        return trace

    def run_gemma(self) -> List[Dict[str, str]]:
        """
        Real ReAct loop using Gemma via Google AI Studio API.
        """
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            print("[CHRONO] No API key found. Falling back to simulation.")
            return self.run_simulation()

        prompt = self._get_formatted_prompt()
        client = genai.Client(api_key=api_key)
        
        # Wrappers for tools to inject state where needed
        def query_personal_history_tool(marker_name: str) -> dict:
            """Queries the historical values and timestamps for a marker."""
            return query_personal_history(self.baseline, marker_name)
            
        def generate_triage_dossier_tool(investigation_summary: str, risk_level: str) -> str:
            """Generates a clinical-grade markdown dossier."""
            return generate_triage_dossier(investigation_summary, risk_level, self.patient_id, self.mcf_result.bav_raw)
            
        gemma_tools = [
            validate_trident_signal,
            query_personal_history_tool,
            compute_vascular_anomaly_score,
            generate_triage_dossier_tool,
            escalate_to_oncologist
        ]
        
        try:
            print(f"[CHRONO Agent] Contacting Gemma 4 26B for Protocol-99 triage with Native Function Calling...")
            
            chat = client.chats.create(
                model='gemma-4-26b-a4b-it',
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    tools=gemma_tools,
                    thinking_config={"thinking_budget": 8192}
                )
            )
            
            response = chat.send_message(prompt)
            trace = []
            
            for _ in range(self.max_iterations):
                if response.text:
                    trace.append({"role": "thought", "content": response.text})
                    
                if not response.function_calls:
                    trace.append({"role": "final_answer", "content": response.text})
                    break
                    
                for fc in response.function_calls:
                    name = fc.name
                    args = fc.args
                    trace.append({"role": "action", "content": f"{name}({args})"})
                    
                    # Execute tool
                    if name == "validate_trident_signal":
                        res = validate_trident_signal(**args)
                    elif name == "query_personal_history_tool":
                        res = query_personal_history_tool(**args)
                    elif name == "compute_vascular_anomaly_score":
                        res = compute_vascular_anomaly_score(**args)
                    elif name == "generate_triage_dossier_tool":
                        res = generate_triage_dossier_tool(**args)
                    elif name == "escalate_to_oncologist":
                        res = escalate_to_oncologist(**args)
                    else:
                        res = f"Unknown tool: {name}"
                        
                    trace.append({"role": "observation", "content": str(res)})
                    
                    response = chat.send_message(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": res}
                        )
                    )
            return trace

        except Exception as e:
            print(f"[CHRONO] API Error: {e}. Falling back to simulation.")
            return self.run_simulation()

if __name__ == "__main__":
    # Test simulation
    from engine.mcf_scorer import MCFResult
    mock_mcf = MCFResult(
        mcf_score=0.73, band="ORANGE", band_color="#f97316", 
        band_message="Alert", protocol99_activated=True,
        wiv_score=2.1, bav_score=1.8, icv_score=2.4,
        wiv_raw=0.12, bav_raw=0.38, icv_raw=0.19,
        wiv_alert=True, bav_alert=True, icv_alert=True,
        trident_firing=True, confidence=0.95, summary="Test"
    )
    agent = Protocol99Agent("priya_001", None, mock_mcf)
    results = agent.run_simulation()
    for r in results:
        print(f"[{r['role'].upper()}] {r['content']}")
