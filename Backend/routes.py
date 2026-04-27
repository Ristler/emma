from flask import request, jsonify, Response
from services.promptService import (
    generate_tokens,
    trim_turns,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_MAX_CONTEXT_TURNS,
)

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
            num_tokens  (int, optional)  - max tokens to generate; defaults to DEFAULT_MAX_NEW_TOKENS
            temperature (float, optional)- sampling temperature; defaults to DEFAULT_TEMPERATURE
            top_k       (int, optional)  - top-k candidate pool size; defaults to DEFAULT_TOP_K
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
        max_tokens   = int(data.get("num_tokens", DEFAULT_MAX_NEW_TOKENS))
        temperature  = float(data.get("temperature", DEFAULT_TEMPERATURE))
        top_k        = int(data.get("top_k", DEFAULT_TOP_K))
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

            return Response(_generate(), mimetype="text/plain")
        else:
            reply = generate_tokens(
                turns_for_model,
                num_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            turns.extend([prompt, reply])
            _sessions[session_id] = trim_turns(turns, DEFAULT_MAX_CONTEXT_TURNS)
            return jsonify({"generated_text": reply})
