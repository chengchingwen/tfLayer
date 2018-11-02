import tensorflow as tf

from .base_layer import Layer
from .layer import PWFFN

class BiRNN(Layer):
    def __init__(self, ff, fcell, bcell, act=tf.nn.tanh, scope="BiRNN", reuse=None):
        super().__init__(scope, reuse)
        self.fcell = fcell
        self.bcell = bcell
        self.ff = ff
        self.act = act

    def prehook(self, x):
        inputs = super().prehook(x)
        if len(x) == 1:
            return inputs[0], None
        else:
            return inputs

    def _build(self):
        self.pwffn = PWFFN(self.ff, act=self.act, reuse=self._reuse)

    def call(self, x, length):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fcell, self.bcell,
                                                          x,
                                                          sequence_length=length,
                                                          dtype=tf.float32,
                                                          parallel_iterations=100,
        )
        return self.pwffn(tf.concat(outputs, -1)), self.pwffn(tf.concat(states, -1))


class FuseBiLSTM(Layer):
    def __init__(self, hidden, act=tf.nn.tanh, ff=1024, scope="FuseBiLSTM", reuse=None):
        super().__init__(scope, reuse)
        self.hidden = hidden
        self.act = act
        self.ff = ff

    def prehook(self, x):
        inputs = super().prehook(x)
        if len(x) == 1:
            return inputs[0], None
        else:
            return inputs

    def _build(self):
        self.fw = tf.nn.rnn_cell.LSTMCell(self.hidden)
        self.bw = tf.nn.rnn_cell.LSTMCell(self.hidden)
        self.birnn = BiRNN(self.ff, self.fw, self.bw, self.act, reuse=self._reuse)

    def call(self, x, length):
        return self.birnn(x, length)

