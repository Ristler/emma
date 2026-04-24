from flask import request, jsonify
from model import model
from utils import preprocess

def register_routes(app):

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()

        text = data.get("text")
        if not text:
            return jsonify({"error": "Missing text"}), 400

        x = preprocess(text)
        pred = model.predict(x)

        return jsonify({"prediction": pred.tolist()})