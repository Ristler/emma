import sentencepiece as spm
from tensorflow import keras

from layers import TokenAndPositionEmbedding, TransformerBlock

model = keras.models.load_model(

    "./models/best_emma.keras",

    custom_objects={

        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,

    },
    compile=False,

)

sp = spm.SentencePieceProcessor()
sp.load("./tokenizers/emma_tokenizer.model")