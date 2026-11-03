
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env. "
        "Please check your .env file."
    )

client = genai.Client(api_key=api_key)


def generate_customer_insights(reviews):
    """Generate customer feedback insights using Gemini AI."""

    if not reviews:
        return "No customer reviews available for analysis."

    review_text = "\n".join(
        f"- {str(review)}" for review in reviews[:30]
    )

    prompt = f"""
You are a customer feedback analyst.

Analyze the following real customer reviews:

{review_text}

Provide:
1. A short summary of customer feedback.
2. The main recurring complaints.
3. Positive feedback, if present.
4. Three practical business recommendations.

Use only evidence from the supplied reviews.
Do not invent facts, statistics, or customer experiences.
Clearly mention when the evidence is insufficient.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return "Gemini returned an empty response. Please try again."

    except Exception as error:
        return (
            f"Gemini analysis failed.\n\n"
            f"Error type: {type(error).__name__}\n\n"
            f"Error details: {str(error)[:500]}"
        )
