"""
CHRONO — API Connectivity Test
Verifies that the Google AI Studio API key is working and can reach Gemma.
"""
import os
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
    
    from google import genai
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemma-4-e4b',
            contents='Hello Gemma, confirm you are Protocol-99 ready.'
        )
        print("✅ Success! Gemma responded:")
        print(f"---\n{response.text}\n---")
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")

if __name__ == "__main__":
    test_gemma_connectivity()
