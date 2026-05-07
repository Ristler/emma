"""Generation logic for the GPT2 (LoRA-finetuned) model.

Owns the plain-text User/Assistant prompt format, the HuggingFace
`model.generate()` call, role-bleed cleanup, and threaded streaming via
`TextIteratorStreamer`. Knows nothing about the Keras model.
"""

import re
from threading import Thread

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from model import get_gpt2_tokenizer


# ── Output cleanup helpers ───────────────────────────────────────────────────

def _strip_role_bleed(reply: str) -> str:
    """Trim accidental next-turn markers from generated assistant text."""
    # If the reply itself starts with "Assistant:", drop that leading marker
    # before doing the split-on-Assistant logic below (otherwise we'd return "").
    leading = reply.lstrip()
    if leading.startswith("Assistant:"):
        reply = leading[len("Assistant:"):]

    if "User:" in reply:
        reply = reply[:reply.index("User:")]
    if "Assistant:" in reply:
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


class _StopOnRoleMarker(StoppingCriteria):
    """Halt generation as soon as 'User:' or 'Assistant:' appears in the new text."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        gen_ids = input_ids[0][self.prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return "User:" in text or "Assistant:" in text


# ── Prompt building ──────────────────────────────────────────────────────────

def _build_prompt(conversation_turns: list[str]) -> str:
    """Build the User/Assistant prompt exactly as used during finetuning."""
    prompt = ""
    for i, turn in enumerate(conversation_turns):
        role = "User" if i % 2 == 0 else "Assistant"
        prompt += f"{role}: {turn}\n"
    prompt += "Assistant:"
    return prompt


def _build_prompt_within_budget(
    conversation_turns: list[str],
    tokenizer,
    max_input_tokens: int,
) -> str:
    """Build the prompt, dropping the oldest turns until it fits the budget.

    Always preserves the most recent user turn (the last entry) so the model
    has the current question to respond to.
    """
    turns = list(conversation_turns)
    while True:
        prompt = _build_prompt(turns)
        token_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_len <= max_input_tokens or len(turns) <= 1:
            return prompt
        # Drop the oldest turn (and its paired reply if present) to free budget.
        turns = turns[2:] if len(turns) >= 3 else turns[1:]


# ── Streaming ────────────────────────────────────────────────────────────────

def _stream(
    model,
    tokenizer,
    inputs,
    prompt_len: int,
    *,
    num_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
):
    """Yield text deltas from GPT2 generation as tokens are produced."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stopping = StoppingCriteriaList([_StopOnRoleMarker(tokenizer, prompt_len)])

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=num_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping,
    )

    def _run_generate():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    thread = Thread(target=_run_generate, daemon=True)
    thread.start()

    def _delta_stream():
        # Hold back enough characters to detect a partial role marker before emitting.
        # Longest marker ("Assistant:") is 10 chars; 12 gives a safety margin.
        HOLDBACK = 12
        buffer = ""
        try:
            for new_text in streamer:
                buffer += new_text

                # Truncate cleanly if a full role marker has appeared.
                cut = -1
                for marker in ("User:", "Assistant:"):
                    idx = buffer.find(marker)
                    if idx != -1 and (cut == -1 or idx < cut):
                        cut = idx
                if cut != -1:
                    safe = buffer[:cut].rstrip()
                    if safe:
                        yield safe
                    return

                if len(buffer) > HOLDBACK:
                    emit = buffer[:-HOLDBACK]
                    buffer = buffer[-HOLDBACK:]
                    yield emit

            if buffer:
                cleaned = _trim_noisy_tail(buffer)
                if cleaned:
                    yield cleaned
        finally:
            thread.join()

    return _delta_stream()


# ── Public entry point ───────────────────────────────────────────────────────

def generate(
    model,
    conversation_turns: list[str],
    *,
    num_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    stream: bool = False,
):
    """Generate a reply with the GPT2 finetuned model.

    When `stream=False`, returns the full reply string. When `stream=True`,
    returns a generator yielding text deltas.
    """
    tokenizer = get_gpt2_tokenizer()

    # Reserve room for the new tokens so the prompt + generation never
    # exceeds the model's context window.
    context_window = getattr(tokenizer, "model_max_length", 1024)
    safety_margin = 8
    max_input_tokens = max(64, context_window - num_tokens - safety_margin)

    prompt = _build_prompt_within_budget(
        conversation_turns, tokenizer, max_input_tokens
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    if stream:
        return _stream(
            model, tokenizer, inputs, prompt_len,
            num_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

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

    return reply
