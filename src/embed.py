import tensorflow as tf

from .base_layer import Layer
from .misc import random_embedding

class Embed(Layer):
    def __init__(self,
                 embedding_size=300,
                 vocab=None,
                 weights=None,
                 trainable=True,
                 zeropad=True,
                 scope="Embed", reuse=None):
        super().__init__(scope, reuse)
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.weights = weights
        self.trainable = trainable
        self.zeropad = zeropad

        if self.weights is not None:
            self.embedding_size = self.weights.shape[1]

    def _build(self):
        if self.weights is None:
            self.weights = random_embedding(
                len(self.vocab),
                self.embedding_size
            )

        vs, dim = self.weights.shape
        if self.zeropad:
            shape = (vs-1, dim)
            winit = tf.constant_initializer(self.weights[1:, :])
        else:
            shape = (vs, dim)
            winit = tf.constant_initializer(self.weights)

        self.embedw = tf.get_variable(
            name=f"embedding_wight",
            shape=shape,
            initializer=winit,
            trainable=self.trainable,
            dtype=tf.float32
        )

        if self.zeropad:
            zp = tf.zeros((1, dim), dtype=tf.float32)
            self.embedw = tf.concat((zp, self.embedw), 0)

    def call(self, x):
        return tf.nn.embedding_lookup(self.embedw, x)


