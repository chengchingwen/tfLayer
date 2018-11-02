import tensorflow as tf
import numpy as np

def random_embedding(size, dim, keep_zeros=[]):
    emb = np.random.randn(size, dim)
    for i in keep_zeros:
        emb[i] = np.zeros(dim)

    return emb


def ZeroPadEmbeddingWeight(weights, name="", trainable=True):
    vs, dims = weights.shape
    weight_init = tf.constant_initializer(weights[1:, :])
    embedding_weights = tf.get_variable(
        name=f'{name + "_" if name else ""}embedding_weights', shape=(vs-1, dims),
        initializer=weight_init,
        trainable=trainable)
    zeropad = tf.zeros((1,dims), dtype=tf.float32)
    return tf.concat((zeropad, embedding_weights), 0)

def embedded(weights, name="", trainable=True, mask_padding=True):
    if mask_padding:
        embedding_weights = ZeroPadEmbeddingWeight(weights, name=name, trainable=trainable)
    else:
        weight_init = tf.constant_initializer(weights)
        embedding_weights = tf.get_variable(
            name = f'{name + "_" if name else ""}embedding_weights',
            shape = weights.shape,
            initializer = weight_init,
            trainable = trainable)

    def lookup(x):
        nonlocal embedding_weights
        return tf.nn.embedding_lookup(embedding_weights, x)

    return lookup


def char_conv(inp,
              filter_size=5,
              channel_out=100,
              strides=[1, 1, 1, 1],
              padding="SAME",
              dilations=[1, 1, 1, 1]):
    inc = inp.get_shape().as_list()[-1]
    filts = tf.get_variable("char_filter", shape=(1, filter_size, inc, channel_out), dtype=tf.float32)
    bias = tf.get_variable("char_bias", shape=(channel_out,), dtype=tf.float32)
    conv = tf.nn.conv2d(inp, filts,
                        strides=strides,
                        padding=padding,
                        dilations=dilations) + bias
    out = tf.reduce_max(tf.nn.relu(conv), 2)
    return out

def mask(x, x_mask=None):
    if x_mask is None:
        return x

    dim = x.get_shape().as_list()[-1]
    mask = tf.tile(tf.expand_dims(x_mask, -1), [1, 1, dim])
    return x * mask

def blocksoftmax(x):
    shape = tf.shape(x) #[b, (d1, d2, d3), m]
    b = shape[0]
    c = shape[-1]
    rx = tf.reshape(x, (b, -1, c)) #[b, d1*d2*d3, m]
    sx = tf.nn.softmax(rx, 1)
    return tf.reshape(x, shape)

def batch_op(func, *x, dtype=tf.float32, parallel_iterations=10):
    return tf.map_fn(
        lambda x: func(*x),
        x,
        dtype=dtype,
        parallel_iterations=parallel_iterations
    )

