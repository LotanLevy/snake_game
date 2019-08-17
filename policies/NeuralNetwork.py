
import tensorflow as tf
import numpy as np



class NeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_layers, session=None,name_prefix="", input_=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name_prefix = name_prefix

        self.weights = []
        self.biases = []

        self.session = tf.Session() if session is None else session

        self.layers = self.build_input_layer(input_)

        for i, width in enumerate(hidden_layers):
             self.layers.append(self.build_hidden_layer(i, width, self.layers))

        self.output = self.build_output_layer(self.layers)

        self.probabilities = tf.nn.softmax(self.output, name="{}probabilities".format(self.name_prefix))
        self.output_max = tf.reduce_max(self.output, axis=1)
        self.output_argmax = tf.argmax(self.output, axis=1)


    def build_input_layer(self, input):
        if input is None:
            self.input = tf.placeholder(tf.float32, shape=(None, self.input_dim),
                                        name="{}input".format(self.name_prefix))
        else:
            self.input = input

        return [self.input]

    def build_hidden_layer(self, layer_id, layer_width, layers):
        """

        :param layer_id: a string representation of this layer (in order to identify this layer)
        :param layer_width:  The length of the layer output
        :param layers: the models layers
        :return: An hidden layer
        """
        return self.get_affine_layer("{}hidden{}".format(self.name_prefix, layer_id), layers[-1], layer_width)

    def build_output_layer(self, layers):
        """
        :param layers: the models layers
        :return: The output layer of this model
        """
        return self.get_affine_layer("{}output".format(self.name_prefix), layers[-1], self.output_dim, relu=False)


    def get_affine_layer(self, name_scope, input_tensor, out_channels, relu=True):
        """
        Creates affine layer by operating affine function on the input tensor.
        If relu parameter is true, it'll operate relu function on the affine output.
        :param name_scope: the name of this layer
        :param input_tensor: the previous layer in this network
        :param out_channels: the length of the output
        :param relu: if it true the function will operate relu function on the affine operation layer
        :return: the output of this layer (operates affine function on the output, and relu if the relu parameter is true).
        """
        input_shape = input_tensor.get_shape().as_list()
        input_channels = input_shape[-1]
        with tf.variable_scope(name_scope):
            W = tf.get_variable("weights",
                                initializer=tf.truncated_normal(
                                    [input_channels, out_channels],
                                    stddev=1.0 / np.sqrt(float(input_channels))
                                ))
            b = tf.get_variable("biases",
                                initializer=tf.zeros([out_channels]))

            self.weights.append(W)
            self.biases.append(b)

            A = tf.matmul(input_tensor, W) + b

            if relu:
                R = tf.nn.relu(A)
                return R
            else:
                return A

    def vars_generator(self):
        """Iterate over all the variables of the network."""
        for w in self.weights:
            yield w
        for b in self.biases:
            yield b

    def take(self, indices):
        """Return an operation that takes values from network outputs.
        e.g. NN.predict_max() == NN.take(NN.predict_argmax())
        """
        mask = tf.one_hot(indices=indices, depth=self.output_dim, dtype=tf.bool,
            on_value=True, off_value=False, axis=-1)
        return tf.boolean_mask(self.output, mask)

    def assign(self, other):
        """Return a list of operations that copies other network into self."""
        ops = []
        for (vh, v) in zip(self.vars_generator(), other.vars_generator()):
            ops.append(tf.assign(vh, v))
        return ops

    def init_variables(self):
        self.session.run(tf.global_variables_initializer())


    @staticmethod
    def run_op_in_batches(session, op, batch_dict={}, batch_size=None,
                          extra_dict={}):

        """Return the result of op by running the network on small batches of
        batch_dict."""

        if batch_size is None:
            return session.run(op, feed_dict={**batch_dict, **extra_dict})

        # Probably the least readable form to get an arbitrary item from a dict
        n = len(next(iter(batch_dict.values())))

        s = []
        for i in range(0, n, batch_size):
            bd = {k: b[i: i + batch_size] for (k, b) in batch_dict.items()}
            s.append(session.run(op, feed_dict={**bd, **extra_dict}))

        if s[0] is not None:
            if np.ndim(s[0]):
                return np.concatenate(s)
            else:
                return np.asarray(s)

    def predict_max(self, inputs_feed, batch_size=None):
        """Return max on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_max,
                                      feed_dict, batch_size)

    def predict_argmax(self, inputs_feed, batch_size=None):
        """Return argmax on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_argmax,
                                      feed_dict, batch_size)

    def predict_exploration(self, inputs_feed, epsilon=0.1, batch_size=None):
        """Return argmax with probability (1-epsilon), and random value with
        probabilty epsilon."""

        n = len(inputs_feed)
        out = self.predict_argmax(inputs_feed, batch_size)
        exploration = np.random.random(n) < epsilon
        out[exploration] = np.random.choice(self.output_dim, exploration.sum())
        return out