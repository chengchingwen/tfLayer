import tensorflow as tf

from .base_layer import Layer

class Normalize(Layer):
    def __init__(self, epsilon=1e-8, scope="Normalize", reuse=None):
        super().__init__(scope, reuse)
        self.epsilon = epsilon

    def _build(self):
        self.beta = tf.get_variable("beta", initializer=tf.zeros(self.input_size))
        self.gamma = tf.get_variable("gamma", initializer=tf.ones(self.input_size))

    def call(self, x):
        mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
        normalized = (x - mean) / ((variance + self.epsilon) ** (.5))
        return self.gamma * normalized + self.beta

