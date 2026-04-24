import sentencepiece as spm
import tensorflow
import keras

from layers import TokenAndPositionEmbedding, TransformerBlock

model = keras.models.load_model(

    "Backend/models/best_emma.keras",

    custom_objects={

        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,

    },
    compile=False,

)

sp = spm.SentencePieceProcessor()
sp.load("Backend/tokenizers/emma_tokenizer.model")