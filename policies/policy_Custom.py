from policies import Policy as bp
from policies.Memory import Memory
# from policies.NeuralNetwork import NeuralNetwork
import tensorflow as tf
# from policies.Feature1 import Feature1
import numpy as np

EPSILON = 0.5
DISCOUNT_RATE = 0.8
LEARNING_RATE = 0.01
FEATURES_LEVEL = 4

STATE_DIM = 132
MEMORY_SIZE = 20000
BATCH_SIZE = 100

NETWORK_ASSIGNMENT_FREQUENCY = 10000
HIDDEN_LAYERS_WIDTHS = [STATE_DIM]

class Custom(bp.Policy):

    class Feature:
        """ A class that represent the features """
        @staticmethod
        def next_action_feature(head_pos, direction, action, board):
            """
            Checks what is the value in the cell of the next action and return a vector according to he value
            :param head_pos: head position
            :param direction: direction
            :param action: action
            :param board: board
            :return: zeros vector of size (11,) with 1 at the index of the value in the next position
            """
            direct_vec = np.zeros((11,))
            next_direction = bp.Policy.TURNS[direction][action]
            next_pos = head_pos.move(next_direction)
            board_value = board[next_pos[0], next_pos[1]]
            direct_vec[board_value + 1] = 1
            return direct_vec.tolist()

        @staticmethod
        def features_around_pos_with_level(action, head, board, level):
            """
            A function that fill values vector for the features around the snake head position
            :param action: action
            :param head: head
            :param board: board
            :param level: steps to look ahead
            :return: list of the values around the snake head
            """
            head_pos, direction = head
            values = np.zeros((11,))
            Custom.Feature.features_around_pos_with_level_helper(action, head_pos, direction, board, level, values)
            return values.tolist()

        @staticmethod
        def features_around_pos_with_level_helper(action, cur_pos, cur_direction, board, max_level, values):
            """
            Helper function that fill the values vector around the snake head position
            Calculate a features vector with ones in the features that represents objects that located
            in the cells that after max_level steps from the current position.
            :param action: action
            :param cur_pos: cur_pos
            :param cur_direction: cur_direction
            :param board: board
            :param max_level: max steps to look ahead
            :param values: array to fill with the values of the cell around the cur_pos
            """
            next_direction = bp.Policy.TURNS[cur_direction][action]
            next_pos = cur_pos.move(next_direction)
            new_level = max_level - 1
            if new_level == 0:
                values[board[next_pos[0], next_pos[1]] + 1] = 1
                return

            for next_action in bp.Policy.ACTIONS:
                Custom.Feature.features_around_pos_with_level_helper(next_action, next_pos,
                                                                     next_direction, board,
                                                                     new_level, values)

        @staticmethod
        def build_state_vec(state, level=4):
            """
            Creates feature vector according to the given state and action
            :param state: state
            :param level: steps to look ahead
            :return: feature vector
            """
            features_for_pos = []
            board, head = state
            head_pos, direction = head

            # 3 vector, vector for each direction - indicates the cell value after acting in that
            # direction (33 features)
            for action in bp.Policy.ACTIONS:
                features_for_pos += Custom.Feature.next_action_feature(head_pos, direction,
                                                                 action, board)
                if level >= 2:
                    features_for_pos += Custom.Feature.features_around_pos_with_level(action, head, board, 2)
                if level >= 3:
                    features_for_pos += Custom.Feature.features_around_pos_with_level(action, head, board, 3)
                if level >= 4:
                    features_for_pos += Custom.Feature.features_around_pos_with_level(action, head, board, 4)

            return np.reshape(np.array(features_for_pos), (1, len(features_for_pos)))

    class NeuralNetwork:
        """ A class that represent the NN """

        def __init__(self, input_dim, output_dim, hidden_layers, session=None, name_prefix="", input_=None):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.name_prefix = name_prefix

            self.weights = []
            self.biases = []

            self.session = tf.Session() if session is None else session
            self.layers = self.build_input_layer(input_)

            for i, width in enumerate(hidden_layers):
                self.layers.append(self.build_hidden_layer(i, width, self.layers))

            self.output_layer = self.build_output_layer(self.layers)

            # self.probabilities = tf.nn.softmax(self.output_layer, name="{}probabilities".format(self.name_prefix)) ##LOTAN - CANT FIND WHO IS USING IT
            self.output_max = tf.reduce_max(self.output_layer, axis=1)
            self.output_argmax = tf.argmax(self.output_layer, axis=1)

        def build_input_layer(self, input):
            """
             Build input layer
            :param input: input
            :return: input layer
            """
            if input is None:
                self.input = tf.placeholder(tf.float32, shape=(None, self.input_dim), name="{}input".format(self.name_prefix))
            else:
                self.input = input
            return [self.input]

        def build_hidden_layer(self, layer_id, layer_width, layers):
            """
            Build hidden layer
            :param layer_id: a string representation of this layer (in order to identify this layer)
            :param layer_width:  The length of the layer output
            :param layers: the models layers
            :return: An hidden layer
            """
            return self.get_affine_layer("{}hidden{}".format(self.name_prefix, layer_id), layers[-1], layer_width)

        def build_output_layer(self, layers):
            """
            Build output layer
            :param layers: all layers
            :return: output layer
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
                                    initializer=tf.truncated_normal([input_channels, out_channels],
                                                                               stddev=1.0 / np.sqrt(float(input_channels))))
                b = tf.get_variable("biases", initializer=tf.zeros([out_channels]))
                self.weights.append(W)
                self.biases.append(b)
                A = tf.matmul(input_tensor, W) + b
                if relu:
                    return tf.nn.relu(A)
                else:
                    return A

        def vars_generator(self):
            """ A generator that iterate over all the variables of the network """
            for w in self.weights:
                yield w
            for b in self.biases:
                yield b

        def take(self, indices):
            """
            Return an operation that takes values from network outputs
            :param indices: indices for the mask
            """
            mask = tf.one_hot(indices=indices, depth=self.output_dim, dtype=tf.bool, on_value=True, off_value=False, axis=-1)
            return tf.boolean_mask(self.output_layer, mask)

        def assign(self, other):
            """
            Return a list of operations that copies other network into self
            :param other: other network
            :return: list of operations
            """
            operations = []
            for (vh, v) in zip(self.vars_generator(), other.vars_generator()):
                operations.append(tf.assign(vh, v))
            return operations

        def init_variables(self):
            """ Initialize variables """
            self.session.run(tf.global_variables_initializer())


        @staticmethod
        def run_op_in_batches(session, op, batch_dict={}):
            """
            run the session on the input, and return the value of the op layer.
            Return the result of op by running the network on small batches of batch dictionary
            :param session: session
            :param op: the output layer
            :param batch_dict: a dictionary of the input data
            :return: result of op by running the network on the input
            """
            return session.run(op, feed_dict=batch_dict)

            # if batch_size is None:
            #     return session.run(op, feed_dict={**batch_dict, **extra_dict})
            #
            # size = len(next(iter(batch_dict.values())))
            # session_array = []
            # for i in range(0, size, batch_size):
            #     new_dict = {k: b[i: i + batch_size] for (k, b) in batch_dict.items()}
            #     session_array.append(session.run(op, feed_dict={**new_dict, **extra_dict}))
            #
            # if session_array[0] is not None:
            #     if np.ndim(session_array[0]):
            #         return np.concatenate(session_array)
            #     else:
            #         return np.asarray(session_array)

        def predict_max(self, inputs_feed):
            """
            Return max on NN outputs
            :param inputs_feed: the input vector
            :return: argmax on NN outputs
            """

            feed_dict = {self.input: inputs_feed}
            return self.run_op_in_batches(self.session, self.output_max, feed_dict)

        def predict_argmax(self, inputs_feed):
            """
            Return argmax on NN outputs
            :param inputs_feed: the input vector
            :return: argmax on NN outputs
            """

            feed_dict = {self.input: inputs_feed}
            return self.run_op_in_batches(self.session, self.output_argmax, feed_dict)

        ## LOTAN - Can't find where we used it - MAYBE CAN BE ERASED
        # def predict_exploration(self, inputs_feed, epsilon=0.1, batch_size=None):
        #     """ Return argmax with probability (1-epsilon), and random value with probability epsilon """
        #
        #     n = len(inputs_feed)
        #     out = self.predict_argmax(inputs_feed, batch_size)
        #     exploration = np.random.random(n) < epsilon
        #     out[exploration] = np.random.choice(self.output_dim, exploration.sum())
        #     return out


    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['dr'] = float(policy_args['dr']) if 'dr' in policy_args else DISCOUNT_RATE
        policy_args['learning_rate'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        return policy_args

    def init_run(self):
        self.state_dim = STATE_DIM
        self.action_to_Idx = {a:i for i, a in enumerate(bp.Policy.ACTIONS)}
        self.idx_to_action = {i:a for i, a in enumerate(bp.Policy.ACTIONS)}

        self.memory_db = Memory(self.state_dim, MEMORY_SIZE)
        self.n_steps = 0 # Q assignment counter
        self.session = tf.Session()
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="states")
        self.Q = Custom.NeuralNetwork(self.state_dim, len(bp.Policy.ACTIONS), HIDDEN_LAYERS_WIDTHS,
                               session=self.session, name_prefix="", input_=self.states)
        self.Qh = Custom.NeuralNetwork(self.state_dim, len(bp.Policy.ACTIONS), HIDDEN_LAYERS_WIDTHS,
                                session=self.session, name_prefix="qh_", input_=self.states)
        self.actions = tf.placeholder(tf.int32, (None,), "taken_actions")
        self.q_values = self.Q.take(self.actions)
        self.y = tf.placeholder(tf.float32, (None,), name="y")
        self.loss_function = tf.reduce_mean((self.y - self.q_values) ** 2)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
        self.train_op = self.optimizer.minimize(self.loss_function, var_list=list(self.Q.vars_generator()))
        self.assign_ops = self.Qh.assign(self.Q)
        self.Q.init_variables()
        # self.n_iter = 1  ##LOTAN - CANT FIND WHO IS USING IT
        self.epsilon_step = (1 / (self.game_duration - self.score_scope)) * (self.epsilon)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        The function for choosing an action, given current state.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row.
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        if round > 1:
            self.memory_db.remember(Custom.Feature.build_state_vec(prev_state, FEATURES_LEVEL)[None, :],
                             Custom.Feature.build_state_vec(new_state, FEATURES_LEVEL)[None,:],
                                    [self.action_to_Idx[prev_action]], [reward])
        if np.random.rand() < self.epsilon or round == 1:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            input_feed = Custom.Feature.build_state_vec(new_state, FEATURES_LEVEL)
            act_index = self.Q.predict_argmax(input_feed)
            return self.idx_to_action[act_index[0]]

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        A function that learn the custom policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row.
        """
        if self.n_steps % NETWORK_ASSIGNMENT_FREQUENCY == 0:
            self.session.run(self.assign_ops)

        sample = self.memory_db.sample(BATCH_SIZE)
        max_prediction = self.Qh.predict_max(sample.next_state)
        y = sample.reward + (self.dr * max_prediction)

        feed_dict = {self.states: sample.prev_state, self.actions: sample.action, self.y: y}
        self.session.run(self.train_op, feed_dict=feed_dict)
        self.n_steps += 1

        self.epsilon -= self.epsilon_step
        if round >= self.game_duration - self.score_scope:
            self.epsilon = 0






