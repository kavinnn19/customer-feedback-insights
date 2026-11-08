
import json
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()


def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in your .env file."
        )
    return genai.Client(api_key=api_key)


def _model_name():
    return os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")


def _generate_with_retry(client, prompt, config=None):
    """Retry temporary Gemini errors up to three times."""

    for attempt in range(3):
        try:
            kwargs = {
                "model": _model_name(),
                "contents": prompt,
            }

            if config is not None:
                kwargs["config"] = config

            return client.models.generate_content(**kwargs)

        except Exception as exc:
            error_text = str(exc).upper()

            temporary = any(
                term in error_text
                for term in (
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                )
            )

            if not temporary or attempt == 2:
                raise

            time.sleep(2 ** (attempt + 1))


def generate_customer_insights(reviews):
    """Generate evidence-based customer feedback insights."""

    cleaned = [
        str(review).strip()
        for review in reviews
        if str(review).strip()
    ]

    if not cleaned:
        return "No usable reviews were provided for Gemini analysis."

    prompt = (
        "Analyze these customer reviews using only the evidence provided. "
        "Return: (1) a short summary, "
        "(2) recurring complaints, noting if evidence is limited, "
        "(3) positive feedback, and "
        "(4) three practical recommendations linked to the reviews. "
        "Do not describe a single complaint as widespread.\n\nREVIEWS:\n"
        + "\n".join(
            f"{i + 1}. {review[:1200]}"
            for i, review in enumerate(cleaned[:20])
        )
    )

    client = _client()

    try:
        response = _generate_with_retry(client, prompt)
        return response.text or "Gemini returned an empty response."
    finally:
        client.close()


def classify_reviews_with_gemini(reviews, batch_size=10):
    """Classify reviews with sentiment and short explanations."""

    cleaned = [str(review).strip() for review in reviews]

    if not cleaned:
        return []

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    valid_sentiments = {"Positive", "Negative", "Neutral"}
    all_results = []
    client = _client()

    try:
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start:start + batch_size]

            numbered = "\n".join(
                f"{i + 1}. {review[:1000]}"
                for i, review in enumerate(batch)
            )

            prompt = (
                "Classify each customer review as Positive, Negative, "
                "or Neutral. Give a short explanation based only on "
                "the review text. Do not infer unsupported facts. "
                'Return only valid JSON in this format: '
                '{"results":[{"sentiment":"Positive",'
                '"explanation":"short reason"}]}. '
                f"Return exactly {len(batch)} results in the same order.\n"
                f"REVIEWS:\n{numbered}"
            )

            response = _generate_with_retry(
                client,
                prompt,
                config={"response_mime_type": "application/json"},
            )

            raw = response.text or ""

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                start_json = raw.find("{")
                end_json = raw.rfind("}")

                if start_json < 0 or end_json <= start_json:
                    raise RuntimeError(
                        "Gemini returned an invalid JSON response."
                    )

                parsed = json.loads(raw[start_json:end_json + 1])

            rows = (
                parsed.get("results", [])
                if isinstance(parsed, dict)
                else []
            )

            if len(rows) != len(batch):
                raise RuntimeError(
                    f"Gemini returned {len(rows)} results for "
                    f"{len(batch)} reviews. Please try again."
                )

            for row in rows:
                sentiment = str(
                    row.get("sentiment", "Unknown")
                ).strip().title()

                if sentiment not in valid_sentiments:
                    sentiment = "Unknown"

                all_results.append({
                    "sentiment": sentiment,
                    "explanation": str(
                        row.get(
                            "explanation",
                            "No explanation provided."
                        )
                    ).strip(),
                })

        return all_results

    finally:
        client.close()
