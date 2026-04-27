from pathlib import Path

import sentencepiece as spm
import tensorflow
import keras

from layers import TokenAndPositionEmbedding, TransformerBlock

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH     = BASE_DIR / "models" / "best_emma.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizers" / "emma_tokenizer.model"

model = keras.models.load_model(
    str(MODEL_PATH),
    custom_objects={
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
    },
    compile=False,
)

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PATH))