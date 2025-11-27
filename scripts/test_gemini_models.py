"""
Test script to list available Gemini models.

Usage:
    export GOOGLE_API_KEY="your-key"
    python scripts/test_gemini_models.py
"""

import os
import google.generativeai as genai

# Configure API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not set")
    exit(1)

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
