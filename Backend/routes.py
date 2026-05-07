from flask import request, jsonify, Response
from services.promptService import (
    generate_tokens,
    trim_turns,
    get_defaults,
    DEFAULT_MAX_CONTEXT_TURNS,
)
from model import load_model, get_current_model_name, get_available_models

# In-memory store mapping session_id -> list of alternating conversation turns.
# Each session holds the recent history for one chat session (one browser tab).
_sessions: dict[str, list[str]] = {}


def register_routes(app):

    @app.route("/predict", methods=["POST"])
    def predict():
        """Handle a chat message and return Emma's reply.

        Expects a JSON body with:
            prompt      (str, required)  - the user's message
            session_id  (str, optional)  - identifies the conversation; defaults to "default"
            num_tokens  (int, optional)  - max tokens to generate; defaults to active model's setting
            temperature (float, optional)- sampling temperature; defaults to active model's setting
            top_k       (int, optional)  - top-k candidate pool size; defaults to active model's setting
            stream      (bool, optional) - stream response token by token; defaults to False

        The user message is appended to the session history before generation.
        After generation the assistant reply is appended so the next call sees
        the full conversation context.

        Returns a streaming plain-text response when stream=True, otherwise a
        JSON object {"generated_text": "..."}.
        """
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        session_id   = data.get("session_id", "default")
        defaults     = get_defaults()
        max_tokens   = int(data.get("num_tokens",  defaults["num_tokens"]))
        temperature  = float(data.get("temperature", defaults["temperature"]))
        top_k        = int(data.get("top_k",        defaults["top_k"]))
        stream       = bool(data.get("stream", False))

        turns = _sessions.setdefault(session_id, [])
        turns_for_model = trim_turns(turns + [prompt], DEFAULT_MAX_CONTEXT_TURNS)

        if stream:
            def _generate():
                """Stream reply chunks to the client and save both turns to history when done."""
                full = ""
                for chunk in generate_tokens(
                    turns_for_model,
                    num_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    stream=True,
                ):
                    full += chunk
                    yield chunk
                reply = full.strip()
                if reply:
                    turns.extend([prompt, reply])
                    _sessions[session_id] = trim_turns(turns, DEFAULT_MAX_CONTEXT_TURNS)

            response = Response(_generate(), mimetype="text/plain")
            response.headers["X-Model-Name"] = get_current_model_name()
            response.headers["X-Token-Budget"] = str(max_tokens)
            return response
        else:
            reply = generate_tokens(
                turns_for_model,
                num_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            turns.extend([prompt, reply])
            _sessions[session_id] = trim_turns(turns, DEFAULT_MAX_CONTEXT_TURNS)
            return jsonify({
                "generated_text": reply,
                "model_name": get_current_model_name(),
                "token_budget": max_tokens,
            })

    @app.route("/models", methods=["GET"])
    def list_models():
        """Get list of available models and the currently active model.

        Returns a JSON object with:
            available_models (dict) - mapping of model_id to model metadata
            current_model    (str)  - the currently active model id
        """
        available = get_available_models()
        current = get_current_model_name()
        
        return jsonify({
            "available_models": available,
            "current_model": current,
        })

    @app.route("/models/switch", methods=["POST"])
    def switch_model():
        """Switch to a different model.

        Expects a JSON body with:
            model_name (str, required) - the model id to switch to

        Returns a JSON object with:
            success      (bool)  - whether the switch was successful
            current_model (str)  - the now-active model id
            message      (str)   - status message
        """
        data = request.get_json()
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({"error": "Missing model_name"}), 400
        
        try:
            load_model(model_name)
            return jsonify({
                "success": True,
                "current_model": get_current_model_name(),
                "message": f"Successfully switched to {model_name}",
            })
        except ValueError as e:
            return jsonify({
                "error": str(e),
                "current_model": get_current_model_name(),
            }), 400
