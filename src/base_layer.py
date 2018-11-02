import tensorflow as tf

class Layer:
    def __init__(self, scope, reuse, input_size=None):
        self._scope = scope
        self._reuse = reuse
        self.input_size = input_size
        self.build = False

    def call(self, x):
        raise NotImplementedError("call not implemented")

    def prehook(self, x):
        self.input_size = x[0].get_shape().as_list()[-1]
        return x

    def posthook(self, x, y):
        return y

    def _build(self):
        pass

    def __call__(self, *x):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            x = self.prehook(x)
            if not self.build:
                self._build()
                self.build = True

            y = self.call(*x)
            return self.posthook(x, y)

# class StackLayers(Layer):
#     def __init__(self, layers, n_layer, scope="StackLayers", reuse=None):
#         super().__init__(scope, reuse)
#         self.layers = layers
#         self.n_layer = n_layer

#     def _build(self):
#         if issubclass(type(self.layers), Layer):

#         if type(self.layers) is list:
