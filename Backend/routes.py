from click import prompt
from flask import request, jsonify
from model import model, sp
from utils import preprocess
import numpy as np
import services.promptService as promptService
from services.promptService import generate_tokens

#import services.promptService as promptService

#generation_config = default_generation_config or GenerationConfig()

def register_routes(app):

    from flask import Response

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()
        prompt = data.get("prompt")
        max_tokens = int(data.get("num_tokens", 256))
        temperature = float(data.get("temperature", 1.0))
        stream = bool(data.get("stream", False))
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        if stream:
            def generate():
                for token in generate_tokens(prompt, num_tokens=max_tokens, temperature=temperature, stream=True):
                    yield token
            return Response(generate(), mimetype="text/plain")
        else:
            generated_text = generate_tokens(prompt, num_tokens=max_tokens, temperature=temperature)
            return jsonify({
                "generated_text": generated_text
            })