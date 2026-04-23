"""
CHRONO — API Connectivity Test
Verifies that the Google AI Studio API key is working and can reach Gemma.
"""
import os
import requests
import sys
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv

load_dotenv()

def test_gemma_connectivity():
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_AI_API_KEY not found in .env file.")
        return

    print(f"Testing connectivity with key: {api_key[:5]}...{api_key[-4:]}")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-4b-it:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "Hello Gemma, confirm you are Protocol-99 ready."}]}]
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        print("✅ Success! Gemma responded:")
        print(f"---\n{text}\n---")
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")

if __name__ == "__main__":
    test_gemma_connectivity()
