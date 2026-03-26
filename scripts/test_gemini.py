"""Quick test that Gemini API key works."""
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("What is 2+2? Reply with just the number.")
print(f"Response: {response.text}")
print("Gemini API is working!")
