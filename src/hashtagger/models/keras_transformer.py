"""TensorFlow / Keras implementation of the hashtag Transformer (+ RL fine-tuning).

Same architecture and recipe as the PyTorch model in ``transformer.py``, so the two frameworks
can be compared like-for-like:

  raw caption string
    -> keras.layers.TextVectorization  (word vocabulary from TRAINING captions, min freq 2)
    -> token + learned positional embeddings
    -> 2 pre-LayerNorm Transformer encoder blocks (MultiHeadAttention + GELU feed-forward)
    -> masked mean-pool -> Dense(GELU) -> Dropout -> Dense(|vocab|) logits

Supervised stage: ``model.fit`` with BCE, AdamW, warm-up + cosine decay, gradient clipping.
RL stage: a custom ``tf.GradientTape`` loop implementing REINFORCE with a leave-one-out
(RLOO) baseline on a set-F1 reward, plus a BCE anchor.

The exported model takes raw strings, so it can be served directly (TF Serving / TFLite).
"""
from __future__ import annotations

import os
import re
from collections import Counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
from keras import layers  # noqa: E402

_TOK = re.compile(r"[a-z]+|\d+")


def build_vocabulary(texts: list[str], min_freq: int = 2) -> list[str]:
    cnt = Counter(w for t in texts for w in _TOK.findall(t.lower()))
    return sorted(w for w, c in cnt.items() if c >= min_freq)


@keras.saving.register_keras_serializable(package="hashtagger")
def standardize(x):
    """Same rule as the PyTorch tokenizer (regex [a-z]+|\\d+): lower-case, split letters from digits."""
    x = tf.strings.lower(x)
    x = tf.strings.regex_replace(x, r"([a-z])([0-9])", r"\1 \2")
    x = tf.strings.regex_replace(x, r"([0-9])([a-z])", r"\1 \2")
    return tf.strings.regex_replace(x, r"[^a-z0-9]+", " ")


def make_vectorizer(vocabulary: list[str], max_len: int) -> layers.TextVectorization:
    # index 0 = padding, 1 = [UNK]
    return layers.TextVectorization(standardize=standardize, split="whitespace",
                                    vocabulary=vocabulary, output_mode="int", output_sequence_length=max_len)


@keras.saving.register_keras_serializable(package="hashtagger")
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, max_len, d_model, **kw):
        super().__init__(**kw)
        self.vocab_size, self.max_len, self.d_model = vocab_size, max_len, d_model
        self.tok = layers.Embedding(vocab_size, d_model)
        self.pos = layers.Embedding(max_len, d_model)

    def call(self, ids):
        positions = tf.range(tf.shape(ids)[-1])
        return self.tok(ids) + self.pos(positions)[None]

    def get_config(self):
        return {**super().get_config(), "vocab_size": self.vocab_size, "max_len": self.max_len,
                "d_model": self.d_model}


@keras.saving.register_keras_serializable(package="hashtagger")
class EncoderBlock(layers.Layer):
    """Pre-LN Transformer encoder block (same layout as torch norm_first=True)."""

    def __init__(self, d_model, n_heads, dropout, **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads, self.dropout = d_model, n_heads, dropout
        self.ln1, self.ln2 = layers.LayerNormalization(epsilon=1e-5), layers.LayerNormalization(epsilon=1e-5)
        self.att = layers.MultiHeadAttention(n_heads, d_model // n_heads, dropout=dropout)
        self.ff1, self.ff2 = layers.Dense(4 * d_model, activation="gelu"), layers.Dense(d_model)
        self.drop1, self.drop2 = layers.Dropout(dropout), layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        att_mask = tf.logical_and(mask[:, None, :], mask[:, :, None])        # [B, T, T]
        h = self.ln1(x)
        x = x + self.drop1(self.att(h, h, attention_mask=att_mask, training=training), training=training)
        h = self.ln2(x)
        return x + self.drop2(self.ff2(self.ff1(h)), training=training)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "n_heads": self.n_heads, "dropout": self.dropout}


@keras.saving.register_keras_serializable(package="hashtagger")
class HashtagEncoder(keras.Model):
    """ids [B, T] -> logits [B, n_labels]."""

    def __init__(self, vocab_size, n_labels, max_len=32, d_model=192, n_heads=4, n_layers=2, dropout=0.1,
                 prior=None, **kw):
        super().__init__(**kw)
        self.cfg = dict(vocab_size=vocab_size, n_labels=n_labels, max_len=max_len, d_model=d_model,
                        n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.emb = PositionalEmbedding(vocab_size, max_len, d_model)
        self.blocks = [EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.hidden = layers.Dense(d_model, activation="gelu")
        self.drop = layers.Dropout(dropout)
        bias = "zeros" if prior is None else keras.initializers.Constant(
            np.log(np.clip(prior, 1e-4, 1 - 1e-4) / (1 - np.clip(prior, 1e-4, 1 - 1e-4))))
        self.out = layers.Dense(n_labels, bias_initializer=bias)          # output bias = label prior

    def call(self, ids, training=False):
        mask = tf.not_equal(ids, 0)
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x, mask, training=training)
        x = self.norm(x)
        m = tf.cast(mask, x.dtype)[..., None]
        pooled = tf.reduce_sum(x * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)
        return self.out(self.drop(self.hidden(pooled), training=training))

    def get_config(self):
        return dict(self.cfg)


def string_model(vectorizer: layers.TextVectorization, encoder: HashtagEncoder) -> keras.Model:
    """End-to-end model: raw caption strings -> tag probabilities (for export / serving).

    The encoder is copied into a fresh, un-compiled instance so the exported file holds only the
    weights (14 MB) and not the AdamW optimizer state (which would triple its size)."""
    clean = HashtagEncoder(**encoder.get_config())
    clean(tf.zeros((1, encoder.cfg["max_len"]), tf.int32))
    clean.set_weights(encoder.get_weights())
    inp = keras.Input(shape=(), dtype="string", name="caption")
    return keras.Model(inp, keras.ops.sigmoid(clean(vectorizer(inp))), name="hashtagger")


def train_bce(encoder, ids, Y, epochs, lr, batch_size, seed=42, log=print):
    steps = epochs * int(np.ceil(len(ids) / batch_size))
    sched = keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr * 0.04, decay_steps=steps - steps // 10,
                                                   warmup_target=lr, warmup_steps=steps // 10)
    encoder.compile(optimizer=keras.optimizers.AdamW(learning_rate=sched, weight_decay=0.01, clipnorm=1.0),
                    loss=keras.losses.BinaryCrossentropy(from_logits=True))
    ds = (tf.data.Dataset.from_tensor_slices((ids, Y)).shuffle(len(ids), seed=seed, reshuffle_each_iteration=True)
          .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    hist = encoder.fit(ds, epochs=epochs, verbose=0, callbacks=[keras.callbacks.LambdaCallback(
        on_epoch_end=lambda e, logs: log(f"  [keras] epoch {e + 1}/{epochs} bce={logs['loss']:.4f}"))])
    return [float(x) for x in hist.history["loss"]]


def predict_proba(encoder, ids, batch_size=1024) -> np.ndarray:
    return np.concatenate([tf.sigmoid(encoder(ids[b:b + batch_size], training=False)).numpy()
                           for b in range(0, len(ids), batch_size)])


def f1_reward(S, Y):
    hit = tf.reduce_sum(S * Y, -1)
    denom = tf.reduce_sum(S, -1) + tf.reduce_sum(Y, -1)
    return tf.where(hit > 0, 2 * hit / tf.maximum(denom, 1.0), tf.zeros_like(hit))


def rl_finetune(encoder, ids, Y, epochs, lr, batch_size, n_samples=4, bce_anchor=0.5, eval_fn=None,
                eval_every=50, seed=42, log=print) -> dict:
    """REINFORCE + RLOO on a set-F1 reward, written as a custom tf.GradientTape loop."""
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.0, clipnorm=1.0)
    K = n_samples

    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            logits = encoder(x, training=True)                                  # [B, L]
            probs = tf.sigmoid(logits)
            S = tf.cast(tf.random.uniform(tf.concat([[tf.shape(x)[0], K], tf.shape(probs)[1:]], 0))
                        < probs[:, None, :], tf.float32)                         # [B, K, L] sampled tag sets
            r = f1_reward(S, y[:, None, :])                                     # [B, K]
            adv = tf.stop_gradient(r - (tf.reduce_sum(r, 1, keepdims=True) - r) / (K - 1))   # RLOO
            logp = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=S, logits=tf.repeat(logits[:, None, :], K, axis=1)), -1)  # log pi(S)
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
            loss = -tf.reduce_mean(adv * logp) + bce_anchor * bce
        grads = tape.gradient(loss, encoder.trainable_variables)
        opt.apply_gradients(zip(grads, encoder.trainable_variables))
        greedy = f1_reward(tf.cast(probs >= 0.5, tf.float32), y)
        return tf.reduce_mean(r), tf.reduce_mean(greedy)

    hist = {"step": [0], "reward_sample": [float("nan")], "reward_greedy": [float("nan")],
            "val_setf1": [eval_fn() if eval_fn else float("nan")]}
    n = 0
    for ep in range(epochs):
        ds = (tf.data.Dataset.from_tensor_slices((ids, Y)).shuffle(len(ids), seed=seed + ep)
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))
        rs, rg = [], []
        for x, y in ds:
            a, b = step(x, y)
            rs.append(float(a)); rg.append(float(b)); n += 1
            if eval_fn is not None and n % eval_every == 0:
                hist["step"].append(n); hist["val_setf1"].append(eval_fn())
                hist["reward_sample"].append(float(np.mean(rs[-eval_every:])))
                hist["reward_greedy"].append(float(np.mean(rg[-eval_every:])))
        log(f"  [keras-rl] epoch {ep + 1}/{epochs} reward(sample)={np.mean(rs):.4f} reward(greedy)={np.mean(rg):.4f}")
    return hist
