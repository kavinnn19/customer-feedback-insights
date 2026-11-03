
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="""
        Summarize these customer reviews in simple language:
        1. The package arrived damaged.
        2. The product packaging was poor.
        3. Delivery took too long.

        Identify the main issue and give one practical recommendation.
        """
    )

    print("Gemini test successful!")
    print(response.text)

except Exception as error:
    print("Gemini request failed:", type(error).__name__)
    print(str(error))
