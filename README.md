# Fake News Detection System

A beginner-friendly educational project that uses TF-IDF text features and Logistic Regression to classify news text as `real` or `fake`, with a Flask web interface.

## Features
- Paste a news headline or article into a web form.
- Train a text-classification pipeline with scikit-learn.
- View the model's prediction in the browser.
- Print basic test-set evaluation metrics.

## Project structure

```text
fake_news_detection_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── news.csv
├── templates/
│   └── index.html
└── static/
    └── style.css
```

`model.joblib` is generated after training and is intentionally not included.

## Dataset format

Put a labeled CSV file at `data/news.csv` with two columns:

```csv
text,label
"Example news article text goes here",real
"Another example article goes here",fake
```

Use a properly sourced, labeled dataset with enough examples of both classes. The tiny included CSV is only a demonstration dataset to test the code; it is not suitable for meaningful accuracy claims or real-world fact-checking.

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install packages:

```bash
python -m pip install -r requirements.txt
```

5. Train the model:

```bash
python train_model.py
```

6. Start the web app:

```bash
python app.py
```

7. Open `http://127.0.0.1:5000` in your browser.

## Methodology

1. Load labeled news text from CSV.
2. Split examples into training and test sets.
3. Convert text into TF-IDF features.
4. Train a Logistic Regression classifier.
5. Evaluate on held-out test examples.
6. Save the pipeline and use it in the Flask interface.

## Limitations and responsible use

- A model prediction is not proof that a story is true or false.
- Results depend on the dataset, labels, language, and time period.
- Models can learn source-specific or topic-specific patterns instead of truth.
- Verify important claims with reputable independent sources.
- Do not report demo-dataset results as a reliable real-world accuracy score.

## Future improvements

- Train and evaluate on a larger, well-documented dataset.
- Add precision, recall, F1-score, and a confusion matrix.
- Support additional languages.
- Add source links and a human fact-checking workflow.
- Deploy with a production-ready WSGI server and secure configuration.

## License

For educational use. Check the license and terms of any external dataset before redistributing it.
