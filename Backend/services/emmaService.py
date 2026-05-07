"""Generation logic for the custom Keras Emma model.

Owns the SentencePiece-based prompt encoding, the hand-rolled autoregressive
sampling loop, and all repetition controls. Knows nothing about GPT2.
"""

import os
from collections import Counter

import numpy as np

from model import sp

# ── Special token IDs (must match training) ──────────────────────────────────
PAD_ID       = sp.pad_id()
BOS_ID       = sp.bos_id()
EOS_ID       = sp.eos_id()
USER_ID      = sp.piece_to_id("<user>")
ASSISTANT_ID = sp.piece_to_id("<assistant>")
SEP_ID       = sp.piece_to_id("<sep>")

MODEL_MAX_SEQ_LEN = 512

# ── Sampling defaults ────────────────────────────────────────────────────────
DEFAULT_MAX_NEW_TOKENS       = 80
DEFAULT_TEMPERATURE          = 0.5
DEFAULT_TOP_K                = 40
DEFAULT_REPETITION_PENALTY   = 1.15

# Keras-only repetition controls (no GPT2 equivalent).
DEFAULT_FREQUENCY_PENALTY    = 0.08
DEFAULT_RECENT_TOKEN_PENALTY = 0.12
DEFAULT_RECENT_TOKEN_WINDOW  = 14
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


# ── Prompt building ──────────────────────────────────────────────────────────

def _encode_turn(role_id: int, text: str) -> list[int]:
    """Encode a single conversation turn into token IDs.

    Wraps the encoded text with the speaker's role token at the front and a
    <sep> token at the end, matching the format used during training.
    """
    return [role_id] + sp.encode(text, out_type=int) + [SEP_ID]


def build_prompt_ids(conversation_turns: list[str]) -> list[int]:
    """Convert a list of alternating conversation turns into a flat token ID sequence.

    Turns at even indices are treated as user messages, odd indices as assistant
    messages. The sequence starts with BOS and ends with the ASSISTANT_ID token
    to signal that the model should generate the next assistant reply.
    """
    ids = [BOS_ID]
    for i, turn in enumerate(conversation_turns):
        role_id = USER_ID if i % 2 == 0 else ASSISTANT_ID
        ids.extend(_encode_turn(role_id, turn))
    ids.append(ASSISTANT_ID)
    return ids


# ── Sampling helpers ─────────────────────────────────────────────────────────

def _blocked_ngram_tokens(reply_tokens: list[int], n: int) -> set[int]:
    """Return the set of tokens that would complete an already-seen n-gram."""
    if n < 2 or len(reply_tokens) < n - 1:
        return set()
    prefix = tuple(reply_tokens[-(n - 1):])
    blocked = set()
    for start in range(len(reply_tokens) - n + 1):
        ngram = reply_tokens[start: start + n]
        if tuple(ngram[:-1]) == prefix:
            blocked.add(ngram[-1])
    return blocked


def _apply_repetition_controls(
    logits: np.ndarray,
    reply_tokens: list[int],
    repetition_penalty: float,
    frequency_penalty: float,
    recent_token_penalty: float,
    recent_token_window: int,
    no_repeat_ngram_size: int,
) -> np.ndarray:
    """Adjust raw model logits to reduce repetitive output."""
    adj = logits.astype(np.float64).copy()

    for tid in (PAD_ID, USER_ID, ASSISTANT_ID):
        if 0 <= tid < len(adj):
            adj[tid] = -1e10

    if not reply_tokens:
        return adj

    if repetition_penalty > 1.0:
        for tid in set(reply_tokens):
            if 0 <= tid < len(adj):
                adj[tid] = adj[tid] / repetition_penalty if adj[tid] >= 0 else adj[tid] * repetition_penalty

    if frequency_penalty > 0:
        for tid, count in Counter(reply_tokens).items():
            if 0 <= tid < len(adj):
                adj[tid] -= frequency_penalty * count

    if recent_token_penalty > 0 and recent_token_window > 0:
        for tid in reply_tokens[-recent_token_window:]:
            if 0 <= tid < len(adj):
                adj[tid] -= recent_token_penalty

    for tid in _blocked_ngram_tokens(reply_tokens, no_repeat_ngram_size):
        if 0 <= tid < len(adj):
            adj[tid] = -1e10

    return adj


def _sample_next_token(logits: np.ndarray, reply_tokens: list[int], **ctrl) -> int:
    """Sample the next token ID from adjusted logits."""
    rep_ctrl = {k: v for k, v in ctrl.items() if k not in ("temperature", "top_k")}
    logits = _apply_repetition_controls(logits, reply_tokens, **rep_ctrl)
    temperature = ctrl["temperature"]

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    top_k = ctrl["top_k"]

    if top_k > 0:
        k = min(int(top_k), len(logits))
        top_idx = np.argpartition(logits, -k)[-k:]
        top_log = logits[top_idx]
        probs = np.exp(top_log - np.max(top_log))
        probs /= probs.sum()
        return int(np.random.choice(top_idx, p=probs))

    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return int(np.random.choice(np.arange(len(logits)), p=probs))


# ── Public entry point ───────────────────────────────────────────────────────

def generate(
    model,
    conversation_turns: list[str],
    *,
    num_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    recent_token_penalty: float = DEFAULT_RECENT_TOKEN_PENALTY,
    recent_token_window: int = DEFAULT_RECENT_TOKEN_WINDOW,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    stream: bool = False,
):
    """Generate a reply with the Keras Emma model.

    When `stream=False`, returns the full reply string. When `stream=True`,
    returns a generator yielding text deltas (the new characters at each step).
    """
    ctrl = dict(
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        recent_token_penalty=recent_token_penalty,
        recent_token_window=recent_token_window,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    prompt_ids = build_prompt_ids(conversation_turns)
    generated = prompt_ids[:]
    reply_tokens: list[int] = []

    def _token_stream():
        """Run the autoregressive loop, yielding the full decoded reply so far after each token."""
        for _ in range(num_tokens):
            window = generated[-MODEL_MAX_SEQ_LEN:]
            x = window + [PAD_ID] * (MODEL_MAX_SEQ_LEN - len(window))

            x_arr = np.array([x], dtype=np.int32)
            logits = model(x_arr, training=False)[0].numpy()

            last_real_idx = max(i for i, t in enumerate(x) if t != PAD_ID)
            next_logits = logits[last_real_idx]

            next_id = _sample_next_token(next_logits, reply_tokens, **ctrl)

            if next_id in (EOS_ID, SEP_ID):
                break

            generated.append(next_id)
            reply_tokens.append(next_id)

            yield sp.decode(reply_tokens)

    if stream:
        def _delta_stream():
            prev = ""
            for text in _token_stream():
                if text.startswith(prev):
                    delta = text[len(prev):]
                else:
                    delta = text[len(os.path.commonprefix([prev, text])):]
                if delta:
                    yield delta
                prev = text

        return _delta_stream()

    text = ""
    for text in _token_stream():
        pass
    return text.strip()
