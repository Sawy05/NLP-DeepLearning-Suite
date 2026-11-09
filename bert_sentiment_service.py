
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    result = sentiment_model(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)