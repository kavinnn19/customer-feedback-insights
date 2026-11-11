import streamlit as st
import pandas as pd
import altair as alt
import joblib
import re
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from gemini_insights import generate_customer_insights, classify_reviews_with_gemini

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

def show_sentiment_chart(sentiment_counts):
    """Display sentiment counts with consistent positive/negative/neutral colours."""
    chart_df = sentiment_counts.rename_axis("Sentiment").reset_index(name="Reviews")
    sentiment_order = ["Positive", "Negative", "Neutral"]
    chart_df["Sentiment"] = pd.Categorical(
        chart_df["Sentiment"], categories=sentiment_order, ordered=True
    )
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Sentiment:N", sort=sentiment_order, title=None),
        y=alt.Y("Reviews:Q", title="Number of reviews"),
        color=alt.Color(
            "Sentiment:N",
            scale=alt.Scale(
                domain=sentiment_order,
                range=["#16a34a", "#dc2626", "#eab308"],
            ),
            legend=None,
        ),
        tooltip=["Sentiment:N", "Reviews:Q"],
    )
    st.altair_chart(chart, use_container_width=True)


# Upload and analyze a separate CSV without changing the historical dashboard.
st.header("0. Upload New Customer Reviews")
st.write(
    "Upload a CSV containing review text to analyze new feedback with the trained "
    "sentiment model. Your historical dashboard below remains based on the saved dataset."
)
uploaded_file = st.file_uploader(
    "Choose a customer reviews CSV file",
    type=["csv"],
    help="The file should contain a review text column such as Text, ReviewText, Review, or content."
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        if uploaded_df.empty:
            st.warning("The uploaded CSV has no rows.")
        else:
            # Find a usable review-text column, or combine Summary + Text.
            lower_columns = {str(c).strip().lower(): c for c in uploaded_df.columns}
            text_col = next(
                (lower_columns[name] for name in
                 ["reviewtext", "review_text", "text", "review", "content", "body"]
                 if name in lower_columns),
                None
            )
            summary_col = lower_columns.get("summary")
            if text_col is None and summary_col is not None and "text" in lower_columns:
                uploaded_df["__combined_review_text"] = (
                    uploaded_df[summary_col].fillna("").astype(str).str.strip()
                    + " "
                    + uploaded_df[lower_columns["text"]].fillna("").astype(str).str.strip()
                ).str.strip()
                text_col = "__combined_review_text"

            if text_col is None:
                st.error(
                    "I couldn't find a review text column. Please include a column named "
                    "Text, ReviewText, Review, content, or body (optionally with Summary)."
                )
            else:
                uploaded_df["ReviewText"] = uploaded_df[text_col].fillna("").astype(str).str.strip()
                uploaded_df = uploaded_df[uploaded_df["ReviewText"].ne("")].copy()

                if uploaded_df.empty:
                    st.warning("No non-empty review text was found in this CSV.")
                else:
                    st.caption(f"Loaded {len(uploaded_df):,} reviews from the uploaded CSV.")
                    if st.button("Analyze uploaded reviews", key="analyze_uploaded_reviews"):
                        with st.spinner("Predicting sentiment for uploaded reviews..."):
                            uploaded_df["ML Sentiment"] = model.predict(uploaded_df["ReviewText"])
                            if hasattr(model, "predict_proba"):
                                try:
                                    probabilities = model.predict_proba(uploaded_df["ReviewText"])
                                    classes = list(model.classes_)
                                    uploaded_df["ML Confidence"] = [
                                        round(float(max(row)), 4) for row in probabilities
                                    ]
                                except Exception:
                                    uploaded_df["ML Confidence"] = None
                            else:
                                uploaded_df["ML Confidence"] = None

                            issue_patterns = {
                                "Taste & Flavor": r"\btaste\b|\bflavou?r\b|\bbland\b|\bbitter\b|\bsalty\b|\bsweet\b|\bstale\b|\baftertaste\b",
                                "Packaging": r"\bpackag(?:e|ing|ed)\b|\bwrapper\b|\bbox\b|\bseal\b|\bcontainer\b|\bpacket\b",
                                "Product Quality": r"\bquality\b|\bbroken\b|\bdefect(?:ive)?\b|\bnot working\b|\bdoes not work\b|\bpoorly made\b|\bmold\b|\bmould\b|\bspoiled\b|\bexpired\b",
                                "Delivery & Shipping": r"\bdelivery\b|\bshipping\b|\bshipped\b|\blate\b|\barrived\b|\bcourier\b|\bmissing order\b",
                                "Price & Value": r"\bprice\b|\bexpensive\b|\bcost\b|\boverpriced\b|\bvalue\b|\bworth\b|\bcheap\b",
                                "Ingredients & Safety": r"\bingredients?\b|\ballergy\b|\ballergic\b|\bunsafe\b|\bcontaminat(?:ed|ion)\b|\bchemical\b"
                            }

                            def uploaded_categories(review):
                                matches = [
                                    category for category, pattern in issue_patterns.items()
                                    if re.search(pattern, str(review).lower())
                                ]
                                return ", ".join(matches) if matches else "Other / General"

                            def uploaded_explanation(row):
                                sentiment = str(row["ML Sentiment"])
                                categories = row["Issue Category"]
                                if sentiment == "Negative":
                                    return (
                                        "The trained model classified this review as negative. "
                                        f"Matched issue keywords: {categories}. Check the review text to verify."
                                    )
                                if sentiment == "Positive":
                                    return (
                                        "The trained model classified this review as positive. "
                                        "This is a model prediction, so verify it against the review text."
                                    )
                                return (
                                    "The trained model classified this review as neutral. "
                                    "Neutral sentiment can be harder to classify; check the original text."
                                )

                            uploaded_df["Issue Category"] = uploaded_df["ReviewText"].apply(uploaded_categories)
                            uploaded_df["ML Explanation"] = uploaded_df.apply(uploaded_explanation, axis=1)
                            uploaded_df["LLM Sentiment"] = "Not evaluated yet"
                            uploaded_df["LLM Explanation"] = "Click the Gemini comparison button below to evaluate this review."
                            uploaded_df["ML vs LLM"] = "Not compared yet"

                        st.session_state["uploaded_review_results"] = uploaded_df
                        st.success("Uploaded reviews analyzed.")

                    results = st.session_state.get("uploaded_review_results")
                    if results is not None:
                        st.subheader("Detailed Review Analysis")
                        st.caption(
                            "Each row shows the ML sentiment, keyword-based issue category, and an explanation. "
                            "Gemini sentiment predictions are run separately when you click the comparison button."
                        )
                        if st.button(
                            "Run Gemini sentiment predictions and compare",
                            key="run_gemini_row_comparison"
                        ):
                            with st.spinner("Gemini is classifying uploaded reviews in small batches..."):
                                try:
                                    llm_results = classify_reviews_with_gemini(
                                        results["ReviewText"].astype(str).tolist()
                                    )
                                    results["LLM Sentiment"] = [x.get("sentiment", "Unknown") for x in llm_results]
                                    results["LLM Explanation"] = [x.get("explanation", "") for x in llm_results]
                                    results["ML vs LLM"] = results.apply(
                                        lambda row: "DIFFERENT — review manually" if
                                        row["ML Sentiment"] != row["LLM Sentiment"] and
                                        row["LLM Sentiment"] in ["Positive", "Negative", "Neutral"]
                                        else ("AGREE" if row["ML Sentiment"] == row["LLM Sentiment"] else "Not compared"),
                                        axis=1
                                    )
                                    st.session_state["uploaded_review_results"] = results
                                    st.success("Gemini predictions completed for uploaded reviews.")
                                except Exception as exc:
                                    st.error(
                                        "Gemini could not classify the reviews. Check GEMINI_API_KEY and internet access. "
                                        f"Details: {exc}"
                                    )
                        results = st.session_state.get("uploaded_review_results", results)
                        if "ML vs LLM" in results.columns and results["ML vs LLM"].isin(["AGREE", "DIFFERENT — review manually"]).any():
                            comparison_counts = results["ML vs LLM"].value_counts()
                            ca, cb = st.columns(2)
                            ca.metric("ML and Gemini agree", int(comparison_counts.get("AGREE", 0)))
                            cb.metric("Predictions differ", int(comparison_counts.get("DIFFERENT — review manually", 0)))
                            st.subheader("ML vs Gemini Comparison")
                            st.dataframe(
                                results[results["ML vs LLM"] == "DIFFERENT — review manually"]
                                [["ReviewText", "ML Sentiment", "LLM Sentiment", "LLM Explanation", "ML vs LLM"]],
                                use_container_width=True,
                                hide_index=True
                            )
                        st.subheader("Uploaded Review Analysis")
                        total_uploaded = len(results)
                        counts_uploaded = results["ML Sentiment"].value_counts()
                        metric_cols = st.columns(4)
                        metric_cols[0].metric("Total reviews", f"{total_uploaded:,}")
                        for i, sentiment in enumerate(["Positive", "Negative", "Neutral"], start=1):
                            metric_cols[i].metric(sentiment, f"{counts_uploaded.get(sentiment, 0):,}")

                        show_sentiment_chart(counts_uploaded)
                        st.caption(
                            "Colour guide: 🟩 Positive = good feedback | "
                            "🟥 Negative = complaints | "
                            "🟨 Neutral = neither clearly positive nor negative."
                        )

                        def colour_sentiment(value):
                            colours = {
                                "Positive": "background-color: #dcfce7; color: #166534; font-weight: bold",
                                "Negative": "background-color: #fee2e2; color: #991b1b; font-weight: bold",
                                "Neutral": "background-color: #fef3c7; color: #92400e; font-weight: bold",
                            }
                            return colours.get(value, "")

                        sentiment_columns = [
                            col for col in ["ML Sentiment", "LLM Sentiment"]
                            if col in results.columns
                        ]
                        # Put sentiment columns first so the colour-coded results are
                        # immediately visible without horizontal scrolling.
                        remaining_columns = [
                            col for col in results.columns if col not in sentiment_columns
                        ]
                        display_results = results[sentiment_columns + remaining_columns].copy()
                        styled_results = display_results.style.applymap(
                            colour_sentiment, subset=sentiment_columns
                        )
                        st.dataframe(
                            styled_results,
                            use_container_width=True,
                            hide_index=True
                        )
                        st.download_button(
                            "Download uploaded review analysis (CSV)",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="uploaded_customer_feedback_analysis.csv",
                            mime="text/csv",
                            key="download_uploaded_analysis"
                        )
                        report_lines = [
                            "CUSTOMER FEEDBACK MANAGEMENT REPORT",
                            "",
                            f"Total uploaded reviews: {len(results)}",
                            "ML sentiment counts:",
                            results["ML Sentiment"].value_counts().to_string(),
                            "",
                            "Most frequent issue categories:",
                            results["Issue Category"].value_counts().head(10).to_string(),
                        ]
                        if "ML vs LLM" in results.columns and results["ML vs LLM"].isin(["AGREE", "DIFFERENT — review manually"]).any():
                            report_lines.extend([
                                "",
                                "ML vs Gemini comparison:",
                                results["ML vs LLM"].value_counts().to_string(),
                                "",
                                "Reviews where predictions differ (review manually):",
                                results.loc[results["ML vs LLM"] == "DIFFERENT — review manually", ["ReviewText", "ML Sentiment", "LLM Sentiment", "LLM Explanation"]].to_string(index=False),
                            ])
                        st.download_button(
                            "Download management report (TXT)",
                            data="\n".join(report_lines).encode("utf-8"),
                            file_name="customer_feedback_management_report.txt",
                            mime="text/plain",
                            key="download_management_report"
                        )

                        st.subheader("Gemini Summary for Uploaded Reviews")
                        st.caption(
                            "This asks Gemini to summarize a small sample of uploaded reviews. "
                            "It does not create a separate LLM sentiment prediction for every row."
                        )
                        if st.button("Generate Gemini summary for uploaded reviews",
                                     key="gemini_uploaded_summary"):
                            sample_reviews = [
                                str(x).strip() for x in results["ReviewText"].head(20).tolist()
                                if str(x).strip()
                            ]
                            with st.spinner("Gemini is reviewing a sample of uploaded feedback..."):
                                try:
                                    st.markdown(generate_customer_insights(sample_reviews))
                                except Exception as exc:
                                    st.error(
                                        "Gemini could not generate a summary. Check the API key, "
                                        "internet connection, and Gemini configuration. "
                                        f"Details: {exc}"
                                    )
    except Exception as exc:
        st.error(f"Could not read this CSV. Please check that it is a valid CSV file. Details: {exc}")

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
    show_sentiment_chart(counts)

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

    st.subheader("Negative Reviews by Issue Category")

    # These keyword rules group reviews into understandable issue categories.
    # A review can match more than one category.
    category_patterns = {
        "Taste & Flavor": r"\btaste\b|\bflavou?r\b|\bbland\b|\bbitter\b|\bsalty\b|\bsweet\b|\bstale\b|\baftertaste\b",
        "Packaging": r"\bpackag(?:e|ing|ed)\b|\bwrapper\b|\bbox\b|\bseal\b|\bcontainer\b|\bpacket\b",
        "Product Quality": r"\bquality\b|\bbroken\b|\bdefect(?:ive)?\b|\bnot working\b|\bdoes not work\b|\bpoorly made\b|\bmold\b|\bmould\b|\bspoiled\b|\bexpired\b",
        "Delivery & Shipping": r"\bdelivery\b|\bshipping\b|\bshipped\b|\blate\b|\barrived\b|\bcourier\b|\bmissing order\b",
        "Price & Value": r"\bprice\b|\bexpensive\b|\bcost\b|\boverpriced\b|\bvalue\b|\bworth\b|\bcheap\b",
        "Ingredients & Safety": r"\bingredients?\b|\ballergy\b|\ballergic\b|\bunsafe\b|\bcontaminat(?:ed|ion)\b|\bchemical\b"
    }

    negative["ReviewText"] = negative["ReviewText"].fillna("").astype(str)

    def find_issue_categories(review_text):
        text = str(review_text).lower()
        matches = [
            category for category, pattern in category_patterns.items()
            if re.search(pattern, text)
        ]
        return matches if matches else ["Other / General"]

    negative["IssueCategories"] = negative["ReviewText"].apply(
        find_issue_categories
    )

    category_rows = [
        {"Issue category": category, "Matching reviews": len(negative[
            negative["IssueCategories"].apply(lambda items: category in items)
        ])}
        for category in category_patterns
    ]
    other_count = int(
        negative["IssueCategories"].apply(
            lambda items: items == ["Other / General"]
        ).sum()
    )
    category_rows.append({
        "Issue category": "Other / General",
        "Matching reviews": other_count
    })
    category_counts = pd.DataFrame(category_rows).sort_values(
        "Matching reviews", ascending=False
    )

    st.caption(
        "Categories are suggested by matching keywords in negative reviews. "
        "A review can appear in more than one category, so category counts may overlap. "
        "Inspect the supporting reviews before drawing conclusions."
    )
    st.bar_chart(category_counts.set_index("Issue category"))

    st.subheader("Emerging Issues Detection")
    monthly_negative = negative.copy()
    monthly_negative["Month"] = monthly_negative["ReviewDate"].dt.to_period("M").astype(str)
    available_months = sorted(monthly_negative["Month"].dropna().unique())
    if len(available_months) >= 2:
        previous_month, current_month = available_months[-2], available_months[-1]
        previous_rows = monthly_negative[monthly_negative["Month"] == previous_month]
        current_rows = monthly_negative[monthly_negative["Month"] == current_month]
        issue_comparison = []
        for category, pattern in category_patterns.items():
            previous_count = int(previous_rows["ReviewText"].str.contains(pattern, case=False, regex=True, na=False).sum())
            current_count = int(current_rows["ReviewText"].str.contains(pattern, case=False, regex=True, na=False).sum())
            if current_count > previous_count and current_count >= 3:
                issue_comparison.append({
                    "Potential emerging issue": category,
                    f"{previous_month} reviews": previous_count,
                    f"{current_month} reviews": current_count,
                    "Change": current_count - previous_count,
                })
        st.caption(
            f"Comparing {previous_month} with {current_month}. These are potential signals based on keyword counts; manually check the reviews."
        )
        if issue_comparison:
            st.dataframe(
                pd.DataFrame(issue_comparison).sort_values("Change", ascending=False),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No issue categories met the current emerging-issue threshold in this comparison.")
    else:
        st.info("Not enough monthly data to compare emerging issues.")

    st.subheader("Supporting Negative Reviews by Category")
    selected_category = st.selectbox(
        "Choose an issue category",
        category_counts["Issue category"].tolist(),
        key="negative_issue_category"
    )

    evidence = negative[
        negative["IssueCategories"].apply(
            lambda items: selected_category in items
        )
    ].copy()

    query = st.text_input(
        "Search within supporting reviews (optional)",
        placeholder="e.g. damaged, bitter, expensive",
        key="category_review_search"
    )

    if query.strip():
        evidence = evidence[
            evidence["ReviewText"].str.contains(
                query.strip(), case=False, na=False, regex=False
            )
        ]

    st.write(f"Category: **{selected_category}** · Matching reviews: **{len(evidence):,}**")

    st.dataframe(
        evidence[["ReviewDate", "Score", "ReviewText"]].head(100),
        use_container_width=True
    )

    if not evidence.empty:
        csv_data = evidence.drop(columns=["IssueCategories"]).to_csv(
            index=False
        ).encode("utf-8")
        st.download_button(
            "Download supporting reviews for this category (CSV)",
            data=csv_data,
            file_name="supporting_reviews_by_category.csv",
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

# Show actual positive examples so the dashboard explains what customers liked.
st.header("3. Positive Customer Feedback")
positive = reviews[reviews["PredictedSentiment"] == "Positive"].copy()
if not positive.empty:
    st.write(
        "This section shows examples from reviews predicted as positive. "
        "Use the review text to understand what customers liked; these are model predictions."
    )
    positive_text = positive["ReviewText"].fillna("").astype(str)
    positive_words = CountVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=12,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )
    try:
        word_matrix = positive_words.fit_transform(positive_text)
        word_counts = word_matrix.sum(axis=0).A1
        positive_terms = pd.DataFrame({
            "Frequently mentioned words/phrases": positive_words.get_feature_names_out(),
            "Count": word_counts
        }).sort_values("Count", ascending=False).head(10)
        st.caption(
            "These are frequent words/phrases in positive-predicted reviews, not guaranteed themes. "
            "Read the examples below before drawing conclusions."
        )
        st.dataframe(positive_terms, hide_index=True, use_container_width=True)
    except ValueError:
        st.info("There is not enough text to extract common positive-feedback phrases.")
    st.subheader("Examples of positive customer feedback")
    st.dataframe(
        positive[["ReviewDate", "Score", "ReviewText"]].head(20),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No positive reviews were found in this sample.")

st.header("4. Model Reliability")

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

st.header("5. Test New Customer Feedback")

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

st.header("6. Export Analysis")

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
