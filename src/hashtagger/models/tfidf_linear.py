"""Baseline 2 — TF-IDF (word 1-2 grams) + multi-label logistic regression, trained in TensorFlow.

One linear layer with sigmoid outputs trained with BCE = one-vs-rest logistic regression over
all hashtags at once. The TF-IDF features come from scikit-learn (a sparse matrix); the linear
layer, the loss and the optimiser are TensorFlow (`tf.sparse.sparse_dense_matmul`).
"""
from __future__ import annotations

import os
import pickle

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


def _sparse(X) -> tf.SparseTensor:
    X = X.tocoo()
    idx = np.stack([X.row, X.col], 1).astype(np.int64)
    return tf.sparse.reorder(tf.SparseTensor(idx, X.data.astype(np.float32), X.shape))


class TfidfLinear:
    def __init__(self, n_labels: int, max_features: int = 30000, ngram_range=(1, 2)):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=tuple(ngram_range),
                                   sublinear_tf=True, min_df=2)
        self.n_labels = n_labels
        self.W: tf.Variable | None = None
        self.b: tf.Variable | None = None

    def fit(self, captions: list[str], Y: np.ndarray, epochs=5, lr=0.5, batch_size=512, seed=42,
            log=print) -> list[float]:
        X = self.vec.fit_transform(captions).tocsr()
        prior = Y.mean(0).clip(1e-4, 1 - 1e-4)
        self.W = tf.Variable(tf.zeros([X.shape[1], self.n_labels]))          # start from the label prior,
        self.b = tf.Variable(np.log(prior / (1 - prior)).astype(np.float32))  # not from p = 0.5
        # Adagrad as in classic sparse linear models: rare n-gram features get large steps.
        # Written out by hand (accumulator starts at 0, eps outside the sqrt) because the Keras
        # Adagrad defaults (accumulator 0.1, eps inside the sqrt) shrink the first steps of rare
        # features by ~100x. No weight decay: with an adaptive optimiser an L2 term is normalised
        # per coordinate and becomes a full-size step towards zero (see README, lessons learned).
        acc = [tf.Variable(tf.zeros_like(self.W)), tf.Variable(tf.zeros_like(self.b))]

        @tf.function
        def step(xb, yb):
            with tf.GradientTape() as tape:
                logits = tf.sparse.sparse_dense_matmul(xb, self.W) + self.b
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yb, logits=logits))
            for v, a, g in zip([self.W, self.b], acc, tape.gradient(loss, [self.W, self.b])):
                a.assign_add(tf.square(g))
                v.assign_sub(lr * g / (tf.sqrt(a) + 1e-10))
            return loss

        rng = np.random.default_rng(seed)
        hist = []
        for ep in range(epochs):
            perm = rng.permutation(X.shape[0])
            tot = 0.0
            for s in range(0, len(perm), batch_size):
                ix = perm[s:s + batch_size]
                tot += float(step(_sparse(X[ix]), tf.constant(Y[ix], tf.float32))) * len(ix)
            hist.append(tot / len(perm))
            log(f"  [tfidf] epoch {ep + 1}/{epochs} bce={hist[-1]:.4f}")
        return hist

    def predict_proba(self, captions: list[str], batch_size: int = 4096) -> np.ndarray:
        X = self.vec.transform(captions).tocsr()
        out = [tf.sigmoid(tf.sparse.sparse_dense_matmul(_sparse(X[s:s + batch_size]), self.W) + self.b).numpy()
               for s in range(0, X.shape[0], batch_size)]
        return np.concatenate(out)

    def n_params(self) -> int:
        return int(np.prod(self.W.shape) + np.prod(self.b.shape))

    def save(self, path: str) -> None:
        pickle.dump({"vec": self.vec, "W": self.W.numpy(), "b": self.b.numpy(), "n_labels": self.n_labels},
                    open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "TfidfLinear":
        d = pickle.load(open(path, "rb"))
        m = cls(d["n_labels"])
        m.vec, m.W, m.b = d["vec"], tf.Variable(d["W"]), tf.Variable(d["b"])
        return m
