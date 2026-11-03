import streamlit as st
import pandas as pd
import joblib
import re
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from gemini_insights import generate_customer_insights

st.set_page_config(
    page_title="Customer Feedback Insights",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "data/processed_reviews.csv"
MODEL_PATH = "models/sentiment_model.joblib"

st.title("Customer Feedback Insights")
st.write(
    "Analyze customer sentiment, discover recurring concerns, "
    "explore historical trends, and inspect supporting reviews."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reviews():
    columns = ["Id", "Score", "ReviewDate", "ReviewText", "Text"]
    df = pd.read_csv(DATA_PATH, usecols=columns)

    df["ReviewDate"] = pd.to_datetime(
        df["ReviewDate"], errors="coerce"
    )
    df = df.dropna(subset=["ReviewText", "ReviewDate"])

    return df.sample(
        n=min(30000, len(df)),
        random_state=42
    ).reset_index(drop=True)


if not Path(DATA_PATH).exists() or not Path(MODEL_PATH).exists():
    st.error("Dataset or trained model not found. Check your project files.")
    st.stop()

with st.spinner("Loading reviews and model..."):
    model = load_model()
    reviews = load_reviews()

with st.spinner("Analyzing customer feedback..."):
    if "PredictedSentiment" not in reviews.columns:
        reviews["PredictedSentiment"] = model.predict(
            reviews["ReviewText"].fillna("")
        )

st.header("1. Sentiment Overview")

counts = reviews["PredictedSentiment"].value_counts()
total = len(reviews)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Reviews analyzed", f"{total:,}")
c2.metric(
    "Positive",
    f"{counts.get('Positive', 0):,}",
    f"{counts.get('Positive', 0) / total:.1%}" if total else "0%"
)
c3.metric(
    "Negative",
    f"{counts.get('Negative', 0):,}",
    f"{counts.get('Negative', 0) / total:.1%}" if total else "0%"
)
c4.metric(
    "Neutral",
    f"{counts.get('Neutral', 0):,}",
    f"{counts.get('Neutral', 0) / total:.1%}" if total else "0%"
)

left, right = st.columns(2)

with left:
    st.subheader("Sentiment distribution")
    st.bar_chart(counts)

with right:
    st.subheader("Sentiment over time")

    monthly = reviews.copy()
    monthly["Month"] = monthly["ReviewDate"].dt.to_period("M").astype(str)

    trend = pd.crosstab(
        monthly["Month"],
        monthly["PredictedSentiment"]
    ).sort_index()

    st.line_chart(trend)


st.header("2. Recurring Concerns")

negative = reviews[
    reviews["PredictedSentiment"] == "Negative"
].copy()

if not negative.empty:

    st.subheader("Common two-word phrases in negative reviews")

    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(2, 2),
            max_features=1000,
            min_df=3,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
        )

        phrase_matrix = vectorizer.fit_transform(
            negative["ReviewText"].fillna("")
        )

        phrase_counts = phrase_matrix.sum(axis=0).A1
        phrases = vectorizer.get_feature_names_out()

        excluded_terms = {
            "br br", "nbsp nbsp", "amp amp", "quot quot",
            "lt lt", "gt gt", "http http", "www www"
        }

        valid_indices = [
            i for i, phrase in enumerate(phrases)
            if phrase not in excluded_terms
            and not any(
                artifact in phrase.split()
                for artifact in ["nbsp", "quot", "amp", "http", "https", "www"]
            )
        ]

        common_phrases = (
            pd.DataFrame({
                "Phrase": phrases[valid_indices],
                "Occurrences": phrase_counts[valid_indices]
            })
            .sort_values("Occurrences", ascending=False)
            .head(15)
        )

        if not common_phrases.empty:
            st.bar_chart(common_phrases.set_index("Phrase"))
        else:
            st.info("No useful repeated phrases were found.")

        st.caption(
            "Frequent phrases are clues to possible issues, not confirmed "
            "issue categories. Inspect supporting reviews before drawing conclusions."
        )

    except ValueError:
        st.info("Not enough repeated phrases were found in this sample.")

    st.subheader("Frequent individual words in negative reviews")

    stopwords = {
        "the", "and", "for", "was", "this", "that", "with", "but",
        "have", "had", "not", "you", "are", "very", "too", "from",
        "they", "its", "it", "is", "of", "to", "in", "a", "an",
        "on", "my", "as", "be", "at", "or", "so", "if", "were",
        "product", "one", "would", "could", "get", "just", "when",
        "all", "out", "up", "me", "we", "our", "no", "than",
        "nbsp", "amp", "quot", "http", "https", "www"
    }

    def extract_words(text):
        words = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
        return [word for word in words if word not in stopwords]

    word_counts = Counter()

    for review_text in negative["ReviewText"]:
        word_counts.update(extract_words(review_text))

    common_words = pd.DataFrame(
        word_counts.most_common(15),
        columns=["Word", "Occurrences"]
    )

    if not common_words.empty:
        st.bar_chart(common_words.set_index("Word"))

    st.subheader("Supporting negative reviews")

    query = st.text_input(
        "Filter negative reviews by keyword",
        placeholder="e.g. taste, packaging, quality"
    )

    evidence = negative.copy()

    if query.strip():
        evidence = evidence[
            evidence["ReviewText"].str.contains(
                query, case=False, na=False, regex=False
            )
        ]

    st.write(f"Matching reviews: {len(evidence):,}")

    st.dataframe(
        evidence[["ReviewDate", "Score", "ReviewText"]].head(100),
        use_container_width=True
    )

    if not evidence.empty:
        csv_data = evidence.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download supporting reviews (CSV)",
            data=csv_data,
            file_name="supporting_negative_reviews.csv",
            mime="text/csv"
        )

    st.subheader("Gemini AI Customer Insights")
    st.write(
        "Gemini summarizes a small sample of actual negative reviews, "
        "highlights recurring issues, and suggests evidence-based actions."
    )

    if st.button("Generate Gemini Insights", key="generate_gemini_insights"):
        gemini_reviews = (
            negative["ReviewText"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        gemini_reviews = [
            review for review in gemini_reviews.head(20).tolist()
            if review
        ]

        if not gemini_reviews:
            st.warning("No usable negative reviews are available for analysis.")
        else:
            with st.spinner("Gemini is analyzing customer reviews..."):
                insights = generate_customer_insights(gemini_reviews)

            st.markdown(insights)

    st.caption(
        "Gemini analyzes up to 20 negative reviews from the current sample. "
        "Its findings should be checked against the supporting reviews."
    )

    st.subheader("Suggested business actions")

    st.write(
        "- Identify frequent negative phrases and inspect the reviews behind them."
    )
    st.write(
        "- Group similar complaints into issue categories before prioritizing actions."
    )
    st.write(
        "- Compare issue frequency across months to investigate possible changes."
    )

    st.caption(
        "These are investigation steps. Verify issues against customer evidence "
        "before making business decisions."
    )

else:
    st.info("No negative reviews were found in this sample.")

st.header("3. Model Reliability")

st.write(
    "The model was evaluated on a separate test set of 20,000 reviews "
    "that were not used to train it."
)

reliability_data = pd.DataFrame({
    "Sentiment": ["Negative", "Neutral", "Positive"],
    "Precision": [0.69, 0.34, 0.97],
    "Recall": [0.78, 0.60, 0.87],
    "F1-score": [0.73, 0.43, 0.92]
})

st.metric("Test accuracy", "83.69%")

st.subheader("Performance by sentiment")

st.dataframe(
    reliability_data,
    hide_index=True,
    use_container_width=True
)

st.subheader("F1-score comparison")
st.bar_chart(reliability_data.set_index("Sentiment")[["F1-score"]])

st.info(
    "Negative F1-score: 0.73. Neutral F1-score: 0.43. "
    "Positive F1-score: 0.92. Neutral reviews are harder for this model to classify."
)

st.warning(
    "These results measure agreement with labels derived from star ratings "
    "(1–2 Negative, 3 Neutral, 4–5 Positive), not independently verified human "
    "sentiment. Manually inspect sample predictions before relying on them."
)

st.header("4. Test New Customer Feedback")

new_review = st.text_area(
    "Enter a customer review",
    placeholder="Example: The product arrived late and the packaging was damaged."
)

if st.button("Analyze review"):
    if new_review.strip():
        prediction = model.predict([new_review])[0]
        probabilities = model.predict_proba([new_review])[0]
        classes = model.classes_

        st.subheader(f"Predicted sentiment: {prediction}")

        st.write("Model probability estimates:")
        st.bar_chart(pd.Series(probabilities, index=classes))

        st.caption(
            "Probability estimates are not guarantees of correctness. "
            "The model was trained using labels derived from star ratings."
        )
    else:
        st.warning("Please enter a review first.")

st.header("5. Export Analysis")

export_df = reviews.copy()
export_df["ReviewDate"] = export_df["ReviewDate"].astype(str)

st.download_button(
    "Download analyzed reviews (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_feedback_analysis.csv",
    mime="text/csv"
)

st.caption(
    "The dashboard analyzes a random sample of up to 30,000 historical reviews. "
    "Model predictions and frequent phrases should be validated before business decisions."
)
