import os
import re
from collections import Counter

import numpy as np
import sentencepiece as spm
import torch

from model import get_current_model, get_current_model_type, sp, get_gpt2_tokenizer

# ── Special token IDs (must match training) ──────────────────────────────────
PAD_ID       = sp.pad_id()
BOS_ID       = sp.bos_id()
EOS_ID       = sp.eos_id()
USER_ID      = sp.piece_to_id("<user>")
ASSISTANT_ID = sp.piece_to_id("<assistant>")
SEP_ID       = sp.piece_to_id("<sep>")

MODEL_MAX_SEQ_LEN = 512

# ── Defaults (mirror the demo script) ────────────────────────────────────────
DEFAULT_MAX_NEW_TOKENS       = 80
DEFAULT_TEMPERATURE          = 0.5
DEFAULT_TOP_K                = 40
DEFAULT_REPETITION_PENALTY   = 1.15
DEFAULT_FREQUENCY_PENALTY    = 0.08
DEFAULT_RECENT_TOKEN_PENALTY = 0.12
DEFAULT_RECENT_TOKEN_WINDOW  = 14
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_MAX_CONTEXT_TURNS    = 6


# ── Prompt building ───────────────────────────────────────────────────────────

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

    Example for two turns ["Hello", "Hi there"]:
        [BOS, <user>, ...Hello..., <sep>, <assistant>, ...Hi there..., <sep>, <assistant>]
    """
    ids = [BOS_ID]
    for i, turn in enumerate(conversation_turns):
        role_id = USER_ID if i % 2 == 0 else ASSISTANT_ID
        ids.extend(_encode_turn(role_id, turn))
    ids.append(ASSISTANT_ID)
    return ids


def trim_turns(turns: list[str], max_turns: int) -> list[str]:
    """Keep only the most recent `max_turns` turns from the conversation history.

    Prevents the prompt from growing beyond the model's context window over long
    conversations. Returns the list unchanged if it is already within the limit.
    """
    if max_turns <= 0 or len(turns) <= max_turns:
        return turns
    return turns[-max_turns:]


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _blocked_ngram_tokens(reply_tokens: list[int], n: int) -> set[int]:
    """Return the set of tokens that would complete an already-seen n-gram.

    Looks at the last (n-1) generated tokens as a prefix and finds every
    token that completed that same prefix earlier in the reply. Those tokens
    are blocked so the model cannot repeat the same n-gram phrase.
    """
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
    """Adjust raw model logits to reduce repetitive output.

    Applies four independent controls in order:
    - Structural token blocking: sets logits for PAD, <user>, and <assistant>
      to -inf so they can never be sampled as reply content.
    - Repetition penalty: divides (positive) or multiplies (negative) the logit
      of every token that has already appeared anywhere in the reply, making
      previously used tokens less likely.
    - Frequency penalty: subtracts a fixed amount per occurrence for each token,
      so tokens used many times are penalised more heavily than tokens used once.
    - Recent token penalty: subtracts a flat penalty for each token seen within
      the last `recent_token_window` positions, targeting short-range repetition
      such as stuttering or repeated phrases.
    - N-gram blocking: sets logits to -inf for any token that would complete an
      n-gram already produced in the current reply.

    Returns a float64 copy of the logits with all adjustments applied.
    """
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
    """Sample the next token ID from adjusted logits.

    First applies all repetition controls, then either:
    - Returns the argmax token when temperature is 0 (greedy/deterministic).
    - Applies temperature scaling and samples from the top-k candidates
      using a softmax probability distribution.
    """
    rep_ctrl = {k: v for k, v in ctrl.items() if k not in ("temperature", "top_k")}
    logits = _apply_repetition_controls(logits, reply_tokens, **rep_ctrl)
    temperature = ctrl.get("temperature", DEFAULT_TEMPERATURE)

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    top_k = ctrl.get("top_k", DEFAULT_TOP_K)

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


# ── Core generation ───────────────────────────────────────────────────────────


def _strip_role_bleed(reply: str) -> str:
    """Trim accidental next-turn markers from generated assistant text."""
    if "User:" in reply:
        reply = reply[:reply.index("User:")]
    if "Assistant:" in reply:
        # Keep only first assistant segment if model repeats speaker markers.
        first = reply.split("Assistant:")[0]
        if first.strip():
            reply = first
    return reply.strip()


def _trim_noisy_tail(reply: str) -> str:
    """Remove obvious degenerate numbered-dot tails (e.g., '12.... 13.....')."""
    cleaned = reply.strip()
    cleaned = re.sub(r"(?:\s\d+\s*\.{2,}){3,}\s*$", "", cleaned)
    cleaned = re.sub(r"\.{8,}\s*$", "", cleaned)
    return cleaned.strip()


def _looks_complete(text: str) -> bool:
    """Heuristic to decide if a response ends naturally."""
    t = text.strip()
    if not t:
        return False
    return t.endswith((".", "!", "?", ")", "]", "\"", "'"))

def _generate_gpt2(
    conversation_turns: list[str],
    num_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    stream: bool = False,
):
    """Generate reply using GPT2 model with notebook-aligned prompt and decoding."""
    tokenizer = get_gpt2_tokenizer()
    model = get_current_model()

    # Build prompt exactly like training notebook format.
    prompt = ""
    for i, turn in enumerate(conversation_turns):
        role = "User" if i % 2 == 0 else "Assistant"
        prompt += f"{role}: {turn}\n"
    prompt += "Assistant:"

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate with the same style of settings as training.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens after prompt, matching notebook logic.
    prompt_len = inputs["input_ids"].shape[1]
    reply_ids = outputs[0][prompt_len:]
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True)
    reply = _strip_role_bleed(reply)
    reply = _trim_noisy_tail(reply)

    # Dynamic but bounded continuation: if generation likely stopped due to token cap
    # and still looks incomplete, continue with conservative decoding.
    if len(reply_ids) >= num_tokens and not _looks_complete(reply):
        generated = outputs
        extra_budget = 48
        step = 16

        while extra_budget > 0 and not _looks_complete(reply):
            with torch.no_grad():
                continued = model.generate(
                    input_ids=generated,
                    max_new_tokens=min(step, extra_budget),
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_ids = continued[0][generated.shape[1]:]
            if new_ids.numel() == 0:
                break

            extra_budget -= int(new_ids.shape[0])
            new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            new_text = _strip_role_bleed(new_text)
            new_text = _trim_noisy_tail(new_text)

            if not new_text:
                break

            reply = f"{reply} {new_text}".strip()
            generated = continued

    if stream:
        def _delta_stream():
            # Stream as small text chunks for UI typing effect.
            for ch in reply:
                yield ch

        return _delta_stream()

    return reply


def generate_tokens(
    conversation_turns: list[str],
    num_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    recent_token_penalty: float = DEFAULT_RECENT_TOKEN_PENALTY,
    recent_token_window: int = DEFAULT_RECENT_TOKEN_WINDOW,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    stream: bool = False,
):
    """Generate Emma's reply for the current conversation.

    Builds the structured prompt from conversation history, then autoregressively
    samples up to `num_tokens` new tokens. Stops early if the model produces an
    EOS or <sep> token, which signals the natural end of the assistant's turn.

    When `stream=False` (default), waits for the full reply and returns it as a
    plain string.

    When `stream=True`, returns a generator that yields text deltas (the new
    characters added at each step) so the caller can stream output to the client
    token by token.
    """
    # Route to appropriate model implementation
    model_type = get_current_model_type()
    if model_type == "gpt2":
        return _generate_gpt2(
            conversation_turns,
            num_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stream=stream,
        )
    
    # Keras model generation (original implementation)
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
            
            model = get_current_model()
            
            # Keras model: expects numpy array
            x_arr = np.array([x], dtype=np.int32)
            logits = model(x_arr, training=False)[0].numpy()
            
            last_real_idx = max(i for i, t in enumerate(x) if t != PAD_ID)
            next_logits = logits[last_real_idx]

            next_id = _sample_next_token(next_logits, reply_tokens, **ctrl)

            if next_id in (EOS_ID, SEP_ID):
                break

            generated.append(next_id)
            reply_tokens.append(next_id)

            current_text = sp.decode(reply_tokens)
            yield current_text

    if stream:
        prev = ""

        def _delta_stream():
            """Wrap _token_stream to yield only the new characters added at each step."""
            nonlocal prev
            for text in _token_stream():
                delta = text[len(prev):] if text.startswith(prev) else text[len(os.path.commonprefix([prev, text])):]
                if delta:
                    yield delta
                prev = text

        return _delta_stream()
    else:
        text = ""
        for text in _token_stream():
            pass
        return text.strip()
