import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Convolution2D, Flatten, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from policies import base_policy as bp
import time
from keras.layers import Input
import matplotlib.pyplot as plt

EPSILON = 0.8
# BOARD_STATES_NUM = 5

TURNS = {
    'N': {'L': 'W', 'R': 'E', 'F': 'N'},
    'S': {'L': 'E', 'R': 'W', 'F': 'S'},
    'W': {'L': 'S', 'R': 'N', 'F': 'W'},
    'E': {'L': 'N', 'R': 'S', 'F': 'E'}
}

ACTIONS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
}

MOVES = {
    0: 'L',
    1: 'F',
    2: 'R',
}

class DeepAgent(bp.Policy):

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args


    # def __init__(self):
    def init_run(self):
        self.epsilon = 0.7
        self.learning_rate = 0.005
        self.discount_factor = 0.9
        self.r_sum = 0
        self.replay_buffer = []
        self.batch_size = 15
        self.initial_epsilon = self.epsilon
        self.size = 20
        self.game_duration = 60000
        self.state_size = 99
        self.model = self.bulid_neural_model()


    def bulid_neural_model(self):
        """
        :return: our deep network.
        """
        # model = Sequential()
        # model.add(Dense(7, input_dim=self.state_size, activation='linear'))
        # model.add(LeakyReLU(alpha=0.001))
        # model.add(Dense(5, activation='linear'))
        # model.add(LeakyReLU(alpha=0.001))
        # model.add(Dense(3, activation='linear'))
        model = Sequential()
        model.add(Dense(output_dim=250, activation='relu', input_dim=self.state_size))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=180, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=3, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam())
        return model


    def calc_net_input(self, state):
        """
        calculating network input. is is a vector of 103 coordinates:
        33 coordinates for 3 steps if moving R.
        33 coordinates for 3 steps if moving F.
        33 coordinates for 3 steps if moving L.
        4 coordinates for current heads direction.
        (detailed explanation can be found in our PDF)
        :param state: current state
        :return: the feature vector as input for net (in suitable shape...)
        """
        dict = {'E' : 0, 'W': 1, 'N': 2, 'S':3}
        input = np.zeros(99)

        # for each action, build feature vec for 3 steps:
        for i in range(len(bp.Policy.ACTIONS)):
            action = bp.Policy.ACTIONS[i]
            board, head = state
            head_pos, direction = head
            f = np.zeros(33)

            # update object existence:
            new_direction = bp.Policy.TURNS[direction][action]
            next_position = head_pos.move(new_direction)
            f[board[next_position[0], next_position[1]] + 1] = 1
            head_pos = next_position

            for action2 in bp.Policy.ACTIONS:
                new_direction2 = bp.Policy.TURNS[new_direction][action2]
                next_position2 = head_pos.move(new_direction2)
                f[board[next_position2[0], next_position2[1]] + 12] = 1
                head_pos = next_position2

                for action3 in bp.Policy.ACTIONS:
                    new_direction3 = bp.Policy.TURNS[new_direction][action3]
                    next_position3 = head_pos.move(new_direction3)
                    f[board[next_position3[0], next_position3[1]] + 23] = 1

            input[i*33: 33 + i*33] = f

        return input.reshape(1, input.shape[0])


    # def calc_net_input(self, state):
    #     """
    #     calculating network input. is is a vector of 103 coordinates:
    #     33 coordinates for 3 steps if moving R.
    #     33 coordinates for 3 steps if moving F.
    #     33 coordinates for 3 steps if moving L.
    #     (detailed explanation can be found in our PDF)
    #     :param state: current state
    #     :return: the feature vector as input for net (in suitable shape...)
    #     """
    #     input = np.zeros(self.state_size)
    #
    #     # for each action, build feature vec for 3 steps:
    #     for i in range(len(bp.Policy.ACTIONS)):
    #         action = bp.Policy.ACTIONS[i]
    #         board, head = state
    #         head_pos, direction = head
    #         f = np.zeros(30)
    #
    #         # update object existence:
    #         new_direction = bp.Policy.TURNS[direction][action]
    #         next_position = head_pos.move(new_direction)
    #         f[board[next_position[0], next_position[1]] + 1] = 1
    #         head_pos = next_position
    #
    #         for action2 in bp.Policy.ACTIONS:
    #             new_direction2 = bp.Policy.TURNS[new_direction][action2]
    #             next_position2 = head_pos.move(new_direction2)
    #             f[board[next_position2[0], next_position2[1]] + 11] = 1
    #             head_pos = next_position2
    #
    #             for action3 in bp.Policy.ACTIONS:
    #                 new_direction3 = bp.Policy.TURNS[new_direction][action3]
    #                 next_position3 = head_pos.move(new_direction3)
    #                 f[board[next_position3[0], next_position3[1]] + 22] = 1
    #
    #         input[i*10: 10 + i*10] = f
    #
    #     return input.reshape(1, input.shape[0])


    # def learn(self, round, prev_state, prev_action, reward, new_state):
    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round % 1000 == 0:
            self.model.save_weights('model_weights_%d.h5' % round)

        dict = {'L': 0, 'F': 1, 'R': 2}
        self.r_sum += reward
        # learn on batch of (prev state, prev action, reward, next state) from mem:
        new_state = self.calc_net_input(new_state)
        prev_state = self.calc_net_input(prev_state)

        if len(self.replay_buffer) >= self.batch_size:
            batch = []
            indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
            for idx in indices:
                batch.append(self.replay_buffer[idx])
            # add current learn-input to this batch:
            batch.append((prev_state, prev_action, reward, new_state))
            for state_b, action_b, reward_b, next_state_b in batch:
                target = reward_b
                if round == self.game_duration:
                    target = reward_b + self.discount_factor * np.amax(self.model.predict(next_state_b, steps=1)[0])
                # building the target vector: calculating the prediction vec and changing only the one coordinate.
                # now, in fit, the loss will be calculated only for this coordinate (because other coordinates will give
                # zero, those values in both vectors are the same so their substruction gives zero.
                # for example: if prediction = (a,b,c), target_vec=(a,x,c), so MSE=(a-a)^2 +(b-x)^2 +(c-c)^2 = (b-x)^2 )
                target_vec = self.model.predict(state_b, steps=1)
                target_vec[0][dict[action_b]] = target
                self.model.fit(state_b, target_vec, epochs=1, verbose=0)
                action = MOVES[np.argmax(target_vec)]

        else:
            # learn only on this input (not batch):
            target = reward
            if round != self.game_duration:
                # updating epsilon according to game duration:
                prediction = self.model.predict(new_state, steps=1)
                target = reward + self.discount_factor * np.amax(prediction[0])
            vec_tag = self.model.predict(prev_state, steps=1)
            prev_action_idx = np.argmax(vec_tag)
            action = MOVES[prev_action_idx]
            vec_tag[0][prev_action_idx] = target
            self.model.fit(prev_state, vec_tag, epochs=1, verbose=0)


        self.epsilon -= (5 / self.game_duration) * (0.8 * self.initial_epsilon)
        if round >= self.game_duration - 1000:
            # just exploration:
            print("exploration")
            self.epsilon = 0

        print("%d/1000000" % round)
        if round % 50 == 0:
            print(self.epsilon)
        return action


    def save(self, state, action, reward, new_state):
        """
        add (state, action, reward, new_state) to mem
        """
        self.replay_buffer.append((state, action, reward, new_state))


    # def act(self, round, prev_state, prev_action, reward, new_state,):
    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round % 1000 == 0:
            self.model.save_weights('model_weights_%d.h5' % round)

        if round % 10000 == 0:
            self.replay_buffer = []

        self.r_sum += reward
        if prev_state is not None:
            # prev_state = prev_state[0].reshape((1, self.size, self.size, 1))
            # new_state = self.calc_net_input(new_state[0].reshape((1, self.size, self.size)))
            prev_state = self.calc_net_input(prev_state)
        if round == 0 or np.random.rand() < self.epsilon:
            action = MOVES[np.random.randint(3)]
        else:
            actions_output = self.model.predict(prev_state, steps=1)
            # action which maximizes the reward:
            action = MOVES[np.argmax(actions_output)]

        if prev_state is not None:
            new_state = self.calc_net_input(new_state)
            self.save(prev_state, prev_action, reward, new_state)
        return action

