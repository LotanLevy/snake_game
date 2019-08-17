from policies import Policy as bp

import numpy as np
import collections
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import random
from policies.Feature1 import Feature1



EPSILON = 0.8
DISCOUNT_RATE = 0.8
LEARNING_RATE = 0.8

INIT_SAMPLES_NUM = 20000
MAX_SAMPLES_IN_FILE = 50000

ZERO = 0

DATA = "snake_samples.txt"
CATEGORICAL_ACTIONS = np.array(['L', 'R', 'F'])



class Deep(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes and tries to go to the
    closest good-fruit.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['dr'] = float(policy_args['dr']) if 'dr' in policy_args else DISCOUNT_RATE

        return policy_args

    def init_run(self):
        self.memory = collections.deque(maxlen=MAX_SAMPLES_IN_FILE)
        self.rewards = {}
        self.last_board = None
        self.r_sum = 0
        self.model = self.network()

        self.epsilon_step = (1 / (self.game_duration - self.score_scope)) * (self.epsilon)




    def network(self):
        """
        Our neural network
        :return: model
        """
        model = Sequential()
        model.add(Dense(30, input_dim=132, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam())

        print(model.summary())
        return model


    def to_categorical_from_action(self, action):
        """
        Array representation of the given action
        :param action: action
        :return: Array representation
        """
        categorical = np.zeros((3,))
        categorical[np.where(CATEGORICAL_ACTIONS == action)] = 1
        return categorical

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        batch_size = 20
        if len(self.memory) < batch_size:
            return
        self.learn_from_samples(self.memory, batch_size)

        try:
            self.epsilon -= self.epsilon_step
            if round >= self.game_duration - self.score_scope:
                self.epsilon = 0

            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.epsilon = 0

                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward
        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')









    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):



        if round == self.game_duration - 1:
            minibatch = random.sample(self.memory, min(len(self.memory), MAX_SAMPLES_IN_FILE))

        if round > 1:
            self.remember(prev_state, prev_action, reward, new_state)
        if np.random.rand() < self.epsilon or round == 1:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            new_state_head_pos, _ = new_state[1]
            act_values = self.model.predict(Feature1.build_state_vec(new_state))
            return bp.Policy.ACTIONS[np.argmax(act_values[0])]


    def remember(self, state, action, reward, next_state):
        """
        Saves the data of the current iteration
        :param state: state
        :param action: action
        :param reward: reward
        :param next_state: next_state
        :return:
        """
        new_state_head_pos, _ = next_state[1]
        new_state_features = Feature1.build_state_vec(next_state)
        prev_state_features = Feature1.build_state_vec(state)
        self.memory.append((prev_state_features, action, reward, new_state_features))

    def learn_from_samples(self, samples, batch_size):
        """
        The network model learns from random samples
        :param samples: the samples we learn from
        :param batch_size: number of sub-samples from samples
        :return:
        """
        minibatch = random.sample(samples, batch_size)
        for state, action, reward, next_state, in minibatch:
            target = reward + self.dr * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][np.argmax(self.to_categorical_from_action(action))] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)









