# MonkeyPox Sentiment Analyzer

A Streamlit web app that classifies the sentiment of tweets about the 2022 monkeypox outbreak as **positive** or **negative**, using a Linear SVC model trained on about 20,000 tweets.

## About the project

This project looks at public opinion on the monkeypox outbreak through text analysis of Twitter data. It uses word clouds, charts and descriptive statistics to find sentiment, keyword associations and trends, an approach that can be reused for similar public-health crises.

The results can help governments, health agencies, ministries and private organisations understand public opinion on the outbreak and make better-informed decisions, policies and regulations.

### Objectives

1. Extract concise, sentiment-related information from tweets.
2. Build a model that measures the share of positive and negative tweets.
3. Deploy the model in an interactive web app.

## Features

The app has three tabs:

| Tab | What it does |
| --- | --- |
| **Text Prediction** | Paste one or more tweets separated by semicolons (`;`). The app returns each tweet's sentiment with a confidence score and a positive/negative pie chart. |
| **URL Prediction** | Enter a search term and a tweet limit. The app scrapes matching tweets, classifies them, and shows a results table, a pie chart and a word cloud. You can download the results as a CSV file. |
| **About Project** | Background and objectives of the study, loaded from [about.md](about.md). |

## How the model was built

The full workflow is in [the notebook](notebooks/Sentiment_Analysis_on%20_MonkeyPox.ipynb).

1. **Data collection:** about 20,000 English tweets mentioning monkeypox, scraped with [twint](https://github.com/twintproject/twint) in August 2022 ([notebooks/monkeypoxV1.csv](notebooks/monkeypoxV1.csv)).
2. **Labelling:** each tweet was labelled using [TextBlob](https://textblob.readthedocs.io/) polarity scores.
3. **Preprocessing:** lowercasing; replacing URLs, @mentions and emoticons with placeholder tokens; removing non-letters, repeated characters and stopwords; and POS-aware lemmatisation with NLTK.
4. **Vectorisation:** TF-IDF with unigrams and bigrams (141,037 features).
5. **Modelling:** four classifiers were trained on a 95/5 train/test split and compared:

| Model | Accuracy | Weighted F1 |
| --- | --- | --- |
| **Linear SVC** (used in the app) | **0.92** | **0.92** |
| Random Forest | 0.89 | 0.89 |
| Logistic Regression | 0.87 | 0.86 |
| Bernoulli Naive Bayes | 0.75 | 0.64 |

The trained vectoriser and model are saved as pickle files in [model/](model/).

## Project structure

```
├── app.py                 # Streamlit app (UI and tabs)
├── helper.py              # Preprocessing, inference, plotting and scraping helpers
├── about.md               # Content for the "About Project" tab
├── model/
│   ├── Sentiment-SVC.pickle   # Trained Linear SVC model
│   └── vectoriser.pickle      # Fitted TF-IDF vectoriser
├── notebooks/
│   ├── Sentiment_Analysis_on _MonkeyPox.ipynb  # Data analysis and model training
│   └── monkeypoxV1.csv                         # Scraped tweet dataset
├── images/                # App icon
├── .streamlit/config.toml # Streamlit theme
├── .devcontainer/         # Dev Container / GitHub Codespaces setup
└── requirements.txt
```

## Getting started

### Run locally

The pinned dependencies date from 2022, so Python 3.8–3.10 is recommended.

```bash
git clone https://github.com/seyirex/monkeypox-sentiment-analyzer-webapp.git
cd monkeypox-sentiment-analyzer-webapp

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. On first launch it downloads the NLTK data it needs, which can take a moment.

### Run in GitHub Codespaces or a Dev Container

Open the repository in GitHub Codespaces, or in VS Code with the Dev Containers extension. The container installs the requirements and starts the app automatically on port 8501.

## Running the tests

The unit tests in [tests/](tests/) cover the text preprocessing, labelling, inference and app helper functions in [helper.py](helper.py). They don't need network access: `twint` is stubbed out and Streamlit calls are replaced with fakes.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

On a fresh machine, the first run downloads the NLTK data it needs. Tests marked `xfail` document known bugs and will start passing once those bugs are fixed.

## Known limitations

- **URL Prediction may not work.** It depends on twint, which scrapes Twitter (now X) without the official API. Changes to the platform since 2023 have broken twint, so live scraping is likely to fail. Text Prediction works offline and is not affected.
- **Binary output.** The model predicts only positive or negative. Neutral tweets aren't a separate class.
- **Labels come from TextBlob.** The training labels were generated automatically, not annotated by people, so the model inherits TextBlob's biases.

## Tech stack

Python · Streamlit · scikit-learn · NLTK · TextBlob · pandas · NumPy · Matplotlib · Seaborn · WordCloud · twint

## License

Released under the [MIT License](LICENSE).
