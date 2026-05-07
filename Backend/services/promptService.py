"""Public entry point for reply generation.

Snapshots the active model, then dispatches to the appropriate per-model
service. The two services own their own prompt formats, sampling logic,
and streaming mechanisms — this file only routes between them and exposes
the shared defaults that `routes.py` uses.
"""

from model import get_current_model, get_current_model_type
from services import emmaService, gpt2Service


# ── Shared defaults exposed to the API layer ─────────────────────────────────
# Per-model sampling defaults live in their own service modules. The API layer
# resolves them at request time via `get_defaults()` so the active model's own
# defaults apply when the client doesn't override.
DEFAULT_MAX_CONTEXT_TURNS = 6


def get_defaults() -> dict:
    """Return the active model's sampling defaults."""
    if get_current_model_type() == "gpt2":
        svc = gpt2Service
    else:
        svc = emmaService
    return {
        "num_tokens":         svc.DEFAULT_MAX_NEW_TOKENS,
        "temperature":        svc.DEFAULT_TEMPERATURE,
        "top_k":              svc.DEFAULT_TOP_K,
        "repetition_penalty": svc.DEFAULT_REPETITION_PENALTY,
    }


def trim_turns(turns: list[str], max_turns: int) -> list[str]:
    """Keep only the most recent `max_turns` turns from the conversation history.

    Prevents the prompt from growing beyond the model's context window over long
    conversations. Returns the list unchanged if it is already within the limit.
    """
    if max_turns <= 0 or len(turns) <= max_turns:
        return turns
    return turns[-max_turns:]


def generate_tokens(
    conversation_turns: list[str],
    num_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    stream: bool = False,
    **extra_keras_kwargs,
):
    """Generate the next assistant reply using whichever model is currently loaded.

    Snapshots the model at entry so a mid-flight `/models/switch` can't swap it
    out from under an in-progress generation.

    `extra_keras_kwargs` (e.g. `frequency_penalty`, `recent_token_penalty`,
    `recent_token_window`, `no_repeat_ngram_size`) are accepted for the Keras
    backend and silently ignored by the GPT2 backend, which doesn't support them.

    When `stream=False`, returns the full reply string.
    When `stream=True`, returns a generator yielding text deltas.
    """
    model = get_current_model()
    model_type = get_current_model_type()
    svc = gpt2Service if model_type == "gpt2" else emmaService

    if num_tokens is None:
        num_tokens = svc.DEFAULT_MAX_NEW_TOKENS
    if temperature is None:
        temperature = svc.DEFAULT_TEMPERATURE
    if top_k is None:
        top_k = svc.DEFAULT_TOP_K
    if repetition_penalty is None:
        repetition_penalty = svc.DEFAULT_REPETITION_PENALTY

    if model_type == "gpt2":
        return gpt2Service.generate(
            model,
            conversation_turns,
            num_tokens=num_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stream=stream,
        )

    return emmaService.generate(
        model,
        conversation_turns,
        num_tokens=num_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stream=stream,
        **extra_keras_kwargs,
    )
