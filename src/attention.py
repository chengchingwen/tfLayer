import tensorflow as tf

from .base_layer import Layer
from .layer import Dense, FFN, PWFFN
from .norm import Normalize

class Attentive(Layer):
    def __init__(self, attns, hidden, act=tf.nn.tanh, scope="Attentive", reuse=None):
        super().__init__(scope, reuse)
        self.hidden = hidden
        self.attns = attns
        self.act = act
        self.penalty = tf.constant(0, dtype=tf.float32)

    def _build(self):
        self.ffn = FFN(self.hidden, self.attns, self.act, reuse=self._reuse)

    def call(self, x): #[ B, L, d]
        b = tf.shape(x)[0]
        a = tf.nn.softmax(self.ffn(x), 1) #[B, L, r]

        penal = tf.matmul(tf.transpose(a,(0, 2, 1)), a) - tf.eye(self.attns, batch_shape=[b])#[b, r, r]
        self.penalty += tf.reduce_sum(tf.square(penal))

        return tf.matmul(tf.transpose(a, (0, 2, 1)), x) #[B, r, d]

    def Objective(self):
        return self.penalty

class Attention(Layer):
    def __init__(self, scope="Attention", reuse=None):
        super().__init__(scope, reuse)

    def _build(self):
        self.dense = Dense(1, scope="Dense", reuse=self._reuse)

    def call(self, q, k):
        ql = tf.shape(q)[1]
        kl = tf.shape(k)[1]

        qs = tf.tile(tf.expand_dims(q, 2), [1, 1, kl, 1]) #[B, ql, kl, d]
        ks = tf.tile(tf.expand_dims(k, 1), [1, ql, 1, 1]) #[B, ql, kl, d]
        qk = qs * ks

        h = tf.concat([qs, ks, qk], -1)
        e = tf.squeeze(self.dense(h), -1)
        a = tf.nn.softmax(e, -1)
        v = tf.matmul(a, k)
        return v


class MultiHeadAttention(Layer):
    def __init__(self, head, dk, scope="MultiHeadAttention", reuse=None):
        super().__init__(scope, reuse)
        self.head = head
        self.dk = dk

    def _build(self):
        self.Qt = Dense(self.head * self.dk, scope="Qt", reuse=self._reuse)
        self.Kt = Dense(self.head * self.dk, scope="Kt", reuse=self._reuse)
        self.Vt = Dense(self.head * self.dk, scope="Vt", reuse=self._reuse)

    def call(self, q, k, v):
        b = tf.shape(q)[0]
        ql = tf.shape(q)[1]
        kl = tf.shape(k)[1]
        vl = kl

        Q = self.Qt(q)
        K = self.Kt(k)
        V = self.Vt(v)
        hq = tf.reshape(Q, (b*self.head, ql, self.dk))
        hk = tf.reshape(K, (b*self.head, kl, self.dk))
        hv = tf.reshape(V, (b*self.head, vl, self.dk)) #[b*h, vl, dk]

        qkt = tf.matmul(hq, tf.transpose(hk, (0, 2, 1))) / (self.dk ** 0.5)#[b*h, ql, kl=vl]
        a = tf.nn.softmax(qkt, -1)

        hatten = tf.matmul(a , hv) #[b*h, ql, dk]
        atten = tf.reshape(hatten, (b, ql, self.head*self.dk))
        return atten


class Transformer(Layer):
    def __init__(self, head = 8, dk = 64, ff = 2048, scope="transformer", reuse=None):
        super().__init__(scope, reuse)
        self.head = head
        self.dk = dk
        self.ff = ff

    def _build(self):
        self.multihead_attention = MultiHeadAttention(self.head, self.dk, reuse=self._reuse)
        self.dense = Dense(self.input_size, scope="Dense", reuse=self._reuse)
        self.normalize1 = Normalize(scope="norm1", reuse=self._reuse)
        self.pwffn = PWFFN(self.ff, reuse=self._reuse)
        self.normalize2 = Normalize(scope="norm2", reuse=self._reuse)

    def call(self, q, k, v):
        atten = self.dense(self.multihead_attention(q, k, v))
        aoutput = self.normalize1(atten + q)
        output = self.pwffn(aoutput)
        return self.normalize2(aoutput + output)

