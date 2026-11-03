# Customer Feedback Insights

A machine learning project that analyzes customer reviews, predicts sentiment, identifies recurring concerns, and generates AI-powered customer insights.

## Project Overview

Customer Feedback Insights helps understand customer opinions from product reviews. It combines machine learning, data visualization, and Gemini AI to explore positive, negative, and neutral feedback.

## Features

- **Sentiment Overview:** View the distribution of positive, negative, and neutral reviews.
- **Historical Trends:** Explore sentiment patterns over time.
- **Recurring Concerns:** Identify common words and phrases in negative reviews and inspect supporting feedback.
- **Gemini AI Customer Insights:** Summarize sampled customer reviews and generate possible improvement recommendations.
- **Model Reliability:** Review the model's evaluation results.
- **Test New Feedback:** Enter a review and predict its sentiment.
- **Export Analysis:** Download analyzed reviews as a CSV file.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF text features
- Logistic Regression
- Streamlit
- Joblib
- Google Gemini API

## Dataset

The project uses the Amazon Fine Food Reviews dataset.

The original dataset is not included in this GitHub repository because the files are large. Obtain the dataset separately and place the original CSV file at:

`data/Reviews.csv`

The preprocessing script generates:

`data/processed_reviews.csv`

## Sentiment Labels

The review rating is mapped to a sentiment label:

- Ratings 1–2: Negative
- Rating 3: Neutral
- Ratings 4–5: Positive

## Model Performance

The reported test accuracy is **83.69%**.

Reported F1 scores:

- Negative: 0.73
- Neutral: 0.43
- Positive: 0.92

These results are from the project's evaluation. Performance may vary depending on the data split and model version. Neutral reviews are more difficult for the model to classify.

## Project Structure

```text
customer-feedback-insights/
├── app.py
├── gemini_insights.py
├── inspect_data.py
├── preprocess_data.py
├── train_model.py
├── test_gemini.py
├── test_gemini_insights.py
├── models/
│   ├── sentiment_model.joblib
│   └── sentiment_model_70_30.joblib
├── data/
│   └── (place dataset files here; CSV files are excluded from Git)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-feedback-insights
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's URL.

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Create the `data` directory and place the downloaded original dataset at `data/Reviews.csv`.

Run:

```bash
python preprocess_data.py
```

This creates `data/processed_reviews.csv`.

### 5. Configure Gemini AI (optional)

Create a local `.env` file in the project root and add your own Google Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit your real `.env` file or share your API key publicly.

### 6. Run the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Notes

- The dataset CSV files and `.env` file are excluded from Git.
- The dashboard requires the processed dataset and trained model files.
- Gemini-powered insights require a valid API key and internet access.
- Review the supporting customer feedback before acting on AI-generated recommendations.

## Learning Outcomes

This project demonstrates data cleaning, text preprocessing, sentiment classification, model evaluation, interactive dashboard development, and AI-assisted customer feedback analysis.