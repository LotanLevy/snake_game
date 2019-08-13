from policies import Policy as bp
import numpy as np

EPSILON = 0.001
DISCOUNT_RATE = 0.1
LEARNING_RATE = 0.1

FEATURES_NUM = 22

OBJ_RANGE = np.arange(-1, 10)


class Linear(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes according to linear function we
    learned in class
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['dr'] = float(policy_args['dr']) if 'dr' in policy_args else DISCOUNT_RATE
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        return policy_args

    def obj_in_pos(self, pos, direction, obj_num, board):
        """
        Checks if obj_num is around the given position and return 0 or 1 according the result
        :param pos: current position
        :param direction: current direction
        :param obj_num: number of the object
        :param board: board
        :return: 0 or 1
        """
        for d in bp.Policy.ACTIONS:
            new_pos = pos.move(bp.Policy.TURNS[direction][d])
            if board[new_pos[0], new_pos[1]] == obj_num:
                return 1
        return 0

    def features(self, new_state, action):
        """
        Creates feature vector according to the given state and action
        :param new_state:
        :param action:
        :return: feature vector
        """
        features_for_pos = np.zeros(len(OBJ_RANGE) * 2)

        board, head = new_state
        head_pos, direction = head

        next_direction = bp.Policy.TURNS[direction][action]
        next_pos = head_pos.move(next_direction)

        for i in range(len(OBJ_RANGE)):
            features_for_pos[i] = int(board[next_pos[0], next_pos[1]] == OBJ_RANGE[i])
            features_for_pos[i + 11] = self.obj_in_pos(next_pos, next_direction, OBJ_RANGE[i],
                                                       board)

        return features_for_pos

    def init_run(self):
        self.weights = np.random.rand(FEATURES_NUM)
        self.epsilon_step = (1 / (self.game_duration - self.score_scope)) * (self.epsilon)
        self.r_sum = 0


    def get_Q_value(self, state, action):
        """
        Calculates Q value
        :param state:
        :param action:
        :return: Q value
        """
        return np.dot(self.weights, self.features(state, action))

    def get_max_Q_value(self, state):
        """
        Calculates maximum Q value and its action
        :param state:
        :return: action: the action that maximize the Q value
                  max Q value: maximum Q value
        """
        values = []
        for action in bp.Policy.ACTIONS:
            values.append(self.get_Q_value(state, action))

        max_val_index = np.argmax(values)
        return bp.Policy.ACTIONS[max_val_index], values[max_val_index]

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        A function that learn the linear policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row.
        """
        _, q_max = self.get_max_Q_value(new_state)

        delta = self.lr * \
                (reward + self.dr * q_max - self.get_Q_value(prev_state, prev_action)) * \
                self.features(prev_state, prev_action)

        self.weights += delta
        self.epsilon -= self.epsilon_step
        if round >= self.game_duration - self.score_scope:
            self.epsilon = 0

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
        if round == 1 or np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            max_action, q_max = self.get_max_Q_value(new_state)
            return max_action


