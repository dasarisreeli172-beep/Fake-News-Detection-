from pathlib import Path
import joblib
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if not news_text:
            message = "Please enter a headline or news article."
        else:
            model = load_model()
            if model is None:
                message = "The model has not been trained yet. Run: python train_model.py"
            else:
                prediction = str(model.predict([news_text])[0]).upper()
    return render_template(
        "index.html",
        prediction=prediction,
        message=message,
        news_text=news_text
    )

if __name__ == "__main__":
    # Debug mode is for local development only.
    app.run(debug=True)
