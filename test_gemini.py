"""Quick script to test Gemini API and list available models"""
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

print("Available Gemini models that support generateContent:\n")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Description: {model.description}")
        print()
