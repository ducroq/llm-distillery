"""
Test script to list available Gemini models.

Usage:
    python scripts/test_gemini_models.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from ground_truth.secrets_manager import get_secrets_manager

# Configure API using SecretsManager
secrets = get_secrets_manager()
api_key = secrets.get_gemini_key()

if not api_key:
    print("ERROR: Gemini API key not found")
    print("Set GOOGLE_API_KEY environment variable or add to config/credentials/secrets.ini")
    sys.exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models:")
print("=" * 60)

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
            print(f"  Display name: {model.display_name}")
            print(f"  Description: {model.description[:100]}...")
            print()
except Exception as e:
    print(f"ERROR: {e}")

print("=" * 60)
print("\nRecommended model names to try:")
print("  - gemini-1.5-flash-002")
print("  - gemini-1.5-flash")
print("  - models/gemini-1.5-flash-002")
