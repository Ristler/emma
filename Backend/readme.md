# Emma Backend

Flask backend that loads a trained transformer model and serves chat responses via a REST API.

---

## Setup

1. Place the trained model file in `Backend/models/` (e.g. `best_emma.keras`)
2. Place the SentencePiece tokenizer in `Backend/tokenizers/` (e.g. `emma_tokenizer.model`)
3. Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r Backend/requirements.txt
```

4. Start the server from the project root:

```bash
python Backend/app.py
```

The server runs on `http://localhost:5000` by default.

---

## File Structure

```
Backend/
├── app.py                  # Flask app entry point, registers routes
├── routes.py               # HTTP endpoint definitions
├── model.py                # Loads the Keras model and tokenizer at startup
├── layers.py               # Custom Keras layers required to deserialize the model
├── utils.py                # Text preprocessing utilities
├── requirements.txt        # Python dependencies
├── models/                 # Drop .keras model files here
└── services/
    └── promptService.py    # Core generation logic
```

---

## How It Works

### Startup

When the server starts, `model.py` loads two artifacts using absolute paths derived from its own file location (so the server can be started from any directory):

- The Keras model (`models/best_emma.keras`)
- The SentencePiece tokenizer (`tokenizers/emma_tokenizer.model`)

The model and tokenizer are imported as module-level singletons and shared across all requests.

### Prompt Format

Emma was trained on a structured conversation format. Every request is converted into this format before being fed to the model:

```
[BOS] <user> {message} <sep> <assistant> {reply} <sep> ... <assistant>
```

The sequence always ends with the `<assistant>` token to signal that the model should generate the next assistant turn. This is handled by `build_prompt_ids()` in `promptService.py`.

### Conversation History

Each chat session is identified by a `session_id` sent with every request. The backend keeps an in-memory store (`_sessions`) mapping session IDs to their list of alternating user/assistant turns. Up to the 6 most recent turns are included in the prompt to keep the total sequence within the model's 512-token context window. Both the user message and the assistant reply are only committed to history together after a successful response — an aborted or failed generation leaves the history unchanged.

### Token Generation

The model generates one token at a time in an autoregressive loop:

1. The current token sequence is padded to 512 tokens and passed to the model
2. Logits are read from the position of the last real (non-pad) token
3. Several post-processing controls are applied to the logits to reduce repetition:
   - **Repetition penalty** — reduces the probability of any token already present in the reply
   - **Frequency penalty** — penalises tokens proportionally to how many times they have appeared
   - **Recent token penalty** — adds an extra penalty for tokens seen in the last 14 positions
   - **N-gram blocking** — prevents the model from completing a 3-token phrase it has already produced
4. The next token is sampled from the top-40 candidates using temperature scaling (default 0.5)
5. Generation stops when the model produces an `<eos>` or `<sep>` token, or when the token limit is reached

### Streaming

When `stream: true` is set in the request, the backend returns a `text/plain` streaming response. Each iteration of the generation loop decodes the reply tokens so far with SentencePiece, computes the new characters added since the last step, and yields only that delta to the client. This handles SentencePiece's subword merging correctly — partial tokens are not emitted until they are fully resolved.

---

## API

### `POST /predict`

Generate a reply from Emma.

**Request body (JSON):**

| Field        | Type    | Required | Default | Description                          |
|--------------|---------|----------|---------|--------------------------------------|
| `prompt`     | string  | yes      |         | The user's message                   |
| `session_id` | string  | no       | `"default"` | Identifies the conversation session |
| `num_tokens` | int     | no       | `80`    | Maximum tokens to generate           |
| `temperature`| float   | no       | `0.5`   | Sampling temperature                 |
| `top_k`      | int     | no       | `40`    | Top-k sampling pool size             |
| `stream`     | bool    | no       | `false` | Stream response token by token       |

**Response (non-streaming):**
```json
{ "generated_text": "Hello! How can I help you?" }
```

**Response (streaming):**  
`Content-Type: text/plain` — plain text chunks streamed incrementally as the reply is generated.

**Error:**
```json
{ "error": "Missing prompt" }
```
