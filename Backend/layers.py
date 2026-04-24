import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size: int, maxlen: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.d_model = d_model
        self.token_emb = keras.layers.Embedding(vocab_size, d_model, name="token_embedding")
        self.pos_emb = keras.layers.Embedding(maxlen, d_model, name="position_embedding")

    def build(self, input_shape):
        self.token_emb.build(input_shape)
        self.pos_emb.build(input_shape)
        super().build(input_shape)

    def call(self, input_ids):
        positions = tf.range(start=0, limit=tf.shape(input_ids)[-1], delta=1)
        return self.token_emb(input_ids) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "maxlen": self.maxlen,
                "d_model": self.d_model,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="mha",
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="gelu"),
                keras.layers.Dropout(dropout),
                keras.layers.Dense(d_model),
            ],
            name="ffn",
        )
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln1")
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln2")
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        self.att.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False, padding_mask=None):
        seq_len = tf.shape(x)[1]

        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=x.dtype), -1, 0)
        if padding_mask is not None:
            pad = tf.cast(padding_mask[:, tf.newaxis, tf.newaxis, :], dtype=x.dtype)
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :] * pad
        else:
            causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]

        residual = x
        attn_output = self.att(
            query=x,
            value=x,
            key=x,
            attention_mask=causal_mask,
            training=training,
        )
        attn_output = self.drop1(attn_output, training=training)
        attn_output = tf.cast(attn_output, residual.dtype)
        x = self.ln1(residual + attn_output)

        residual = x
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        ffn_output = tf.cast(ffn_output, residual.dtype)
        return self.ln2(residual + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout,
            }
        )
        return config