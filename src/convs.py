import tensorflow as tf

from .base_layer import Layer
from .misc import batch_op

def batchconv(conv, x, f, strides, padding, dilation, parallel_iterations=64):
    """
    x: shape [b, ..., c]
    f: shape [b, ..., ci, co]

    out: shape[b, ..., co]
    """
    x = tf.expand_dims(x, 1)
    C = batch_op(
        lambda x, f: tf.squeeze(
            conv(x, f, strides, padding),
            0
        ),
        x,
        f,
        parallel_iterations=parallel_iterations,
    )

    return C

def batchconv1d(x, filters, strides=[1,1,1], padding="VALID", dilation=[1,1,1], parallel_iterations=64):
    """
    x: shape [b, w, c]
    filters: shape [b, w, ci, co]
    """
    stride = strides[1] #due to tf.nn.conv1d api not consistent
    return batchconv(tf.nn.conv1d, x, filters, stride, padding, dilation, parallel_iterations=parallel_iterations)

def batchconv2d(x, filters, strides=[1,1,1,1], padding="VALID", dilation=[1,1,1,1], parallel_iterations=64):
    """
    x: shape [b, h, w, c]
    filters: shape [b, h, w, ci, co]
    """
    return batchconv(tf.nn.conv2d, x, filters, strides, padding, dilation, parallel_iterations=parallel_iterations)

def batchconv3d(x, filters, strides=[1,1,1,1,1], padding="VALID", dilation=[1,1,1,1,1], parallel_iterations=64):
    """
    x: shape [b, d, h, w, c]
    filters: shape [b, d, h, w, ci, co]
    """
    return batchconv(tf.nn.conv3d, x, filters, strides, padding, dilation, parallel_iterations=parallel_iterations)

def batch_depthwiseconv2d_native(x, filters, strides=[1,1,1,1], padding="VALID", dilation=[1,1,1,1]):
    """
    x: shape [b, h, w, c]
    filters: shape [b, h, w, ci, m]

    output: shape [b ,h, w, ci * m]
    """
    return batchconv(tf.nn.depthwise_conv2d, x, filters, strides, padding, dilation)

def depthwiseconv1d(x, filters, strides=[1,1,1], padding="VALID", dilation=[1,1,1]):
    """
    x: shape [1, w, c]
    filters: shape [w, ci, m]

    output: shape [w, ci * m]
    """
    xt = tf.transpose(x, (2, 1, 0)) #[b=c, w, 1]
    fts = tf.expand_dims(tf.transpose(filters, (1, 0, 2)), -2) #[ci, w, 1, m]
    r = tf.transpose(
        batchconv1d(xt, fts, strides=strides, padding=padding, dilation=dilation, parallel_iterations=100),
        (1, 0, 2)
    ) #r [c, w, m] -> [w, c, m]
    rshape = tf.shape(r)
    return tf.reshape(r, (1, rshape[0], -1))


def depthwiseconv2d(x, filters, strides=[1,1,1,1], padding="VALID", dilation=[1,1,1,1]):
    """
    x: shape [1, h, w, c]
    filters: shape [h, w, ci, m]

    output: shape [h, w, ci * m]
    """
    xt = tf.transpose(x, (3, 1, 2, 0)) #[b=c, h, w, 1]
    fts = tf.expand_dims(tf.transpose(filters, (2, 0, 1, 3)), -2) #[ci, h, w, 1, m]
    r = tf.transpose(
        batchconv2d(xt, fts, strides=strides, padding=padding, dilation=dilation, parallel_iterations=100),
        (1, 2, 0, 3)
    ) #r [c, h, w, m] -> [h, w, c, m]
    rshape = tf.shape(r)
    return tf.reshape(r, (1, rshape[0], rshape[1], -1))


def depthwiseconv3d(x, filters, strides=[1,1,1,1,1], padding="VALID", dilation=[1,1,1,1,1]):
    """
    x: shape [1, d, h, w, c]
    filters: shape [d, h, w, ci, m]

    output: shape [d, h, w, ci * m]
    """
    xt = tf.transpose(x, (4, 1, 2, 3, 0)) #[b=c, d, h, w, 1]
    # fts = tf.expand_dims(tf.expand_dims(tf.transpose(filters, (3, 0, 1, 2)), -1), -1) #[b=ci, d, h, w, 1, 1]
    # return tf.transpose(batchconv3d(xt, fts, strides=strides, padding=padding, dilation=dilation), (4, 1, 2, 3, 0))
    fts = tf.expand_dims(tf.transpose(filters, (3, 0, 1, 2, 4)), -2) #[ci, d, h, w, 1, m]
    r = tf.transpose(
        batchconv3d(xt, fts, strides=strides, padding=padding, dilation=dilation, parallel_iterations=100),
        (1, 2, 3, 0, 4)
    ) #r [c, d, h, w, m] -> [d, h, w, c, m]
    rshape = tf.shape(r)
    return tf.reshape(r, (1, rshape[0], rshape[1], rshape[2], -1))

def batch_depthwiseconv1d(x, filters, strides=[1,1,1], padding="VALID", dilation=[1,1,1]):
    """
    x: shape [b, w, c]
    filters: shape [b, w, ci, m]

    output: shape [b, w, ci * m]
    """
    return batchconv(depthwiseconv1d, x, filters, strides, padding, dilation)

def batch_depthwiseconv2d(x, filters, strides=[1,1,1,1], padding="VALID", dilation=[1,1,1,1]):
    """
    x: shape [b, h, w, c]
    filters: shape [b, h, w, ci, m]

    output: shape [b, h, w, ci * m]
    """
    return batchconv(depthwiseconv2d, x, filters, strides, padding, dilation)

def batch_depthwiseconv3d(x, filters, strides=[1,1,1,1,1], padding="VALID", dilation=[1,1,1,1,1]):
    """
    x: shape [b, d, h, w, c]
    filters: shape [b, d, h, w, ci, m]

    output: shape [b, d, h, w, ci * m]
    """
    return batchconv(depthwiseconv3d, x, filters, strides, padding, dilation)


class BatchConv(Layer):
    _batchconvs = {
        1: batchconv1d,
        2: batchconv2d,
        3: batchconv3d,
    }
    def __init__(self, strides=1, dilation=1, padding="VALID", scope="BatchConv", reuse=None):
        super().__init__(scope, reuse)
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

        if isinstance(self.strides, int):
            self.strides = [self.strides] * self.dim

        if isinstance(self.dilation, int):
            self.dilation = [self.dilation] * self.dim

        assert self.dim == len(self.strides)
        assert self.dim == len(self.dilation)

        self.strides = [1] + self.strides + [1]
        self.dilation = [1] + self.dilation + [1]

        self.conv = self._batchconvs[self.dim]

    def call(self, x, f):
        """
        x: shape [b, ..., c]
        f: shape [b, ..., ci, co]

        out: shape[b, ..., co]
        """

        return self.conv(x, f, strides=self.strides, padding=self.padding, dilation=self.dilation)

class BatchDepthwiseConv(Layer):
    _batchdepthwiseconvs = {
        1: batch_depthwiseconv1d,
        2: batch_depthwiseconv2d_native,
        3: batch_depthwiseconv3d,
    }
    def __init__(self, strides=1, dilation=1, padding="VALID", scope="BatchConv", reuse=None):
        super().__init__(scope, reuse)
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

        if isinstance(self.strides, int):
            self.strides = [self.strides] * self.dim

        if isinstance(self.dilation, int):
            self.dilation = [self.dilation] * self.dim

        assert self.dim == len(self.strides)
        assert self.dim == len(self.dilation)

        self.strides = [1] + self.strides + [1]
        self.dilation = [1] + self.dilation + [1]

        self.conv = self._batchdepthwiseconvs[self.dim]

    def call(self, x, f):
        return self.conv(x, f, strides=self.strides, padding=self.padding, dilation=self.dilation)

    def posthook(self, x, y):
        s0 = x[0].get_shape().as_list() #[b, (...), ci]
        s1 = x[1].get_shape().as_list() #[b, (...), ci, m]
        b, ci, m = s0[0], s1[-2], s1[-1]
        ns = [None] * len(s0)
        ns[0], ns[-1] = b, ci*m
        y.set_shape(ns)
        return y

