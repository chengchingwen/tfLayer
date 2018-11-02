import tensorflow as tf

from .base_layer import Layer

class Dense(Layer):
    def __init__(self, hidden, act=None, scope="Dense", reuse=None):
        super().__init__(scope, reuse)
        self.act = act
        self.hidden = hidden

    def _build(self):
        self.dense = tf.layers.Dense(self.hidden, activation=self.act, _reuse=self._reuse)

    def call(self, x):
        return self.dense(x)


class Conv(Layer):
    _convs = {
        1: tf.layers.Conv1D,
        2: tf.layers.Conv2D,
        3: tf.layers.Conv3D,
    }

    def __init__(self, filter_num, kernel_size, act=None,
                 strides=1,
                 dilation=1,
                 padding="VALID",
                 scope="Conv", reuse=None
    ):
        super().__init__(scope, reuse)
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.act = act
        self.padding = padding
        self.strides = strides
        self.dilation = dilation

    def prehook(self, x):
        inputs = super().prehook(x)
        self.dim = len(inputs[0].get_shape().as_list()) - 2

        return inputs

    def _build(self):
        if self.dim > 3:
            raise Exception("dim > 3 not support")

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * self.dim

        if isinstance(self.strides, int):
            self.strides = [self.strides] * self.dim

        if isinstance(self.dilation, int):
            self.dilation = [self.dilation] * self.dim

        assert self.dim == len(self.kernel_size)
        assert self.dim == len(self.strides)
        assert self.dim == len(self.dilation)

        self.conv = self._convs[self.dim](filters=self.filter_num,
                                          kernel_size=self.kernel_size,
                                          strides=self.strides,
                                          padding=self.padding,
                                          dilation_rate=self.dilation,
                                          activation=self.act,
        )


    def call(self, x):
        return self.conv(x)


class Highway(Layer):
    def __init__(self, act=None, scope="Highway", reuse=None):
        super().__init__(scope, reuse)
        self.act = act

    def _build(self):
        size = self.input_size
        self.kT = Dense(size, tf.nn.sigmoid, scope="Gate", reuse=self._reuse)
        self.kH = Dense(size, self.act, scope="Act", reuse=self._reuse)

    def call(self, x):
        T = self.kT(x)
        H = self.kH(x)
        return T * H + (1.0 - T) * x


class HighwayNetwork(Layer):
    def __init__(self, n_layer, act=None, scope="HighwayNetwork", reuse=None):
        super().__init__(scope, reuse)
        self.n_layer = n_layer
        self.act = act

    def _build(self):
        self.hws = [
            Highway(act = self.act, scope=f"Highway{i+1}", reuse=self._reuse)
            for i in range(self.n_layer)
        ]
        self.pwffn = PWFFN(2048, reuse=self._reuse)

    def call(self, x):
        rh = x
        for hw in self.hws:
            rh = hw(rh)

        return self.pwffn(rh)


class FFN(Layer):
    def __init__(self, dim1, dim2=None, act=None, scope="FFN", reuse=None):
        super().__init__(scope, reuse)
        self.dim1 = dim1
        self.dim2 = dim2 if dim2 is not None else dim1
        self.act = act

    def _build(self):
        self.dense1 = Dense(self.dim1, self.act, scope="dense1", reuse=self._reuse)
        self.dense2 = Dense(self.dim2, scope="dense2", reuse=self._reuse)

    def call(self, x):
        return self.dense2(self.dense1(x))


class PWFFN(Layer):
    def __init__(self, ff, act=tf.nn.relu, scope="PWFFN", reuse=None):
        super().__init__(scope, reuse)
        self.ff = ff
        self.act = act

    def _build(self):
        self.ffn = FFN(self.ff, self.input_size, self.act, reuse=self._reuse)

    def call(self, x):
        return self.ffn(x)
