from pathlib import Path

import sentencepiece as spm
import tensorflow
import keras
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from layers import TokenAndPositionEmbedding, TransformerBlock

BASE_DIR = Path(__file__).resolve().parent

# ── Model paths ──────────────────────────────────────────────────────────────
KERAS_MODEL_PATH = BASE_DIR / "models" / "best_emma.keras"
GPT2_MODEL_PATH = BASE_DIR / "models" / "gpt2_finetuned_final"
TOKENIZER_PATH = BASE_DIR / "tokenizers" / "emma_tokenizer.model"

# ── Tokenizer (shared across models) ─────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PATH))

# ── Available models ─────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "emma": {
        "name": "Emma (Keras)",
        "type": "keras",
        "path": KERAS_MODEL_PATH,
    },
    "gpt2_finetuned": {
        "name": "GPT2 Finetuned",
        "type": "gpt2",
        "path": GPT2_MODEL_PATH,
    },
}

# ── Global model state ───────────────────────────────────────────────────────
_current_model = None
_current_model_name = "emma"


def load_keras_model():
    """Load the Emma Keras model with custom layers."""
    return keras.models.load_model(
        str(KERAS_MODEL_PATH),
        custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
        },
        compile=False,
    )


def load_gpt2_model():
    """Load the GPT2 finetuned model with LoRA adapter."""
    import torch
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_model, str(GPT2_MODEL_PATH))
    model.eval()  # Set to evaluation mode
    
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model


def get_gpt2_tokenizer():
    """Load the GPT2 tokenizer from the finetuned model directory."""
    return AutoTokenizer.from_pretrained(str(GPT2_MODEL_PATH))


def load_model(model_name: str):
    """Load the specified model by name and update global state.
    
    Args:
        model_name: Key from AVAILABLE_MODELS dictionary
        
    Returns:
        The loaded model object
        
    Raises:
        ValueError: If model_name is not recognized
    """
    global _current_model, _current_model_name
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_config = AVAILABLE_MODELS[model_name]
    
    print(f"Loading model: {model_config['name']}...")
    
    if model_config["type"] == "keras":
        _current_model = load_keras_model()
    elif model_config["type"] == "gpt2":
        _current_model = load_gpt2_model()
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    _current_model_name = model_name
    print(f"Model loaded successfully: {model_config['name']}")
    
    return _current_model


def get_current_model():
    """Get the currently loaded model, loading the default if none is loaded yet."""
    global _current_model
    if _current_model is None:
        load_model(_current_model_name)
    return _current_model


def get_current_model_name():
    """Get the name of the currently loaded model."""
    return _current_model_name


def get_current_model_type():
    """Get the type of the currently loaded model."""
    return AVAILABLE_MODELS[_current_model_name]["type"]


def get_available_models():
    """Get metadata about all available models (JSON-serializable format)."""
    return {
        model_id: {
            "name": model_config["name"],
            "type": model_config["type"],
        }
        for model_id, model_config in AVAILABLE_MODELS.items()
    }


# ── Load default model on startup ────────────────────────────────────────────
model = get_current_model()