from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import os

app = Flask(__name__, static_folder=".", template_folder=".")

# Load model and vectorizer once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "models/vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    text = data.get("text", "")
    combined = title + " " + text
    features = vectorizer.transform([combined])
    proba = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]
    label = "True News" if prediction == 1 else "Fake News"
    # proba[1] is probability for True, proba[0] for Fake
    return jsonify(
        {
            "prediction": label,
            "true_proba": float(proba[1]),
            "fake_proba": float(proba[0]),
        }
    )


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(".", "style.css")


if __name__ == "__main__":
    app.run(debug=True)
