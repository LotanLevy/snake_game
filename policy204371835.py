from policies import base_policy as bp
import numpy as np
import keras
import random

EPSILON = 1
LEARNING_RATE = 1
DISCOUNT = 0.4

LEFT=0
RIGHT=1
FORWARD=2

class Costum204371835(bp.Policy):

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = EPSILON
        policy_args['learning_rate'] = LEARNING_RATE
        policy_args['discount'] = DISCOUNT
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.state_size = 33
        self.memory = []
        self.ram = []
        self.action2idx = {'L' : 0, 'R' : 1, 'F' : 2}
        decay = 1 / (self.game_duration - self.score_scope)
        self.decay_epsilon = 5 * EPSILON * decay
        self.model = self._build_model()

    # The model is a sequncial model, with 4 denese layers with leaky relu activation.
    # in the end i return a scalar wchich is the Q(s,a) approximation
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(33, input_dim=self.state_size, activation='linear'))
        model.add(keras.layers.LeakyReLU(alpha=.001))
        model.add(keras.layers.Dense(22, activation='linear'))
        model.add(keras.layers.LeakyReLU(alpha=.001))
        model.add(keras.layers.Dense(11, activation='linear'))
        model.add(keras.layers.LeakyReLU(alpha=.001))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam())
        self.log(model.summary())
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        self.ram.append((state, action, reward, next_state))

    def get_features(self, new_state, action):

        feature_vec = np.zeros(33)
        board, head = new_state
        head_pos, orig_direction = head
        next_position = head_pos.move(bp.Policy.TURNS[orig_direction][action])
        new_direction = bp.Policy.TURNS[orig_direction][action]
        r = next_position[0]
        c = next_position[1]
        #put a '1' in the right feature on first step
        reward_idx = board[r, c] + 1
        feature_vec[reward_idx] = 1
        for a in bp.Policy.ACTIONS:
            new_direction2 = bp.Policy.TURNS[new_direction][a]
            next_position2 = next_position.move(bp.Policy.TURNS[new_direction][a])
            r = next_position2[0]
            c = next_position2[1]
            reward_idx = board[r, c] + 1
            feature_vec[reward_idx + 11] = 1

        for obj in range(-1, 9):
            min_dist = np.inf
            obj_x, obj_y = np.where(board == obj)
            for i in range(len(obj_x)):
                dist = ((obj_x[i] - head_pos[0])**2 +(obj_y[i] - head_pos[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
            feature_vec[obj + 23] = min_dist
            # print("obj idx: {} dist: {}".format(obj, min_dist))

        #     for a2 in bp.Policy.ACTIONS:
        #         # get a Position object of the position in the relevant direction from the head:
        #         new_direction3 = bp.Policy.TURNS[new_direction2][a2]
        #         next_position3 = next_position2.move(bp.Policy.TURNS[new_direction2][a2])
        #         r = next_position3[0]
        #         c = next_position3[1]
        # #put a '1' in the right feature on third step
        #         reward_idx = board[r, c] + 1
        #         feature_vec[reward_idx + 22] = 1
        return feature_vec.reshape(1,33)

    # # given a state and action - returns a vector that represents the board as we showed in pdf
    # def get_features(self, new_state, action):
    #     #init our reprenstation vector:
    #     feature_vec = np.zeros(33)
    #     board, head = new_state
    #     head_pos, orig_direction = head
    #     next_position = head_pos.move(bp.Policy.TURNS[orig_direction][action])
    #     new_direction = bp.Policy.TURNS[orig_direction][action]
    #     r = next_position[0]
    #     c = next_position[1]
    #     #put a '1' in the right feature on first step
    #     reward_idx = board[r, c] + 1
    #     feature_vec[reward_idx] = 1
    #     for a in bp.Policy.ACTIONS:
    #         # get a Position object of the position in the relevant direction from the head:
    #         new_direction2 = bp.Policy.TURNS[new_direction][a]
    #         next_position2 = next_position.move(bp.Policy.TURNS[new_direction][a])
    #         r = next_position2[0]
    #         c = next_position2[1]
    #         reward_idx = board[r, c] + 1
    #     #put a '1' in the right feature on second step
    #         feature_vec[reward_idx + 11] = 1
    #         for a2 in bp.Policy.ACTIONS:
    #             # get a Position object of the position in the relevant direction from the head:
    #             new_direction3 = bp.Policy.TURNS[new_direction2][a2]
    #             next_position3 = next_position2.move(bp.Policy.TURNS[new_direction2][a2])
    #             r = next_position3[0]
    #             c = next_position3[1]
    #     #put a '1' in the right feature on third step
    #             reward_idx = board[r, c] + 1
    #             feature_vec[reward_idx + 22] = 1
    #     return feature_vec.reshape(1,33)

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round <= 2:
            return
        # iterate over last 5 steps that were action and learn them:
        for state, action, reward, next_state, in self.memory:
            q_max = []
            for action_x in bp.Policy.ACTIONS:
                q_max.append(self.model.predict(self.get_features(next_state, action_x))[0])
			# our target to fit our network - we get an vecotr of states and and action (Q,A) and we the network for it.
            target = reward + self.discount * np.max(q_max)
            s_features = self.get_features(state, action)
            target_f = self.model.predict(s_features)
            target_f[0] = target
            self.model.fit(s_features, target_f, epochs=2, verbose=0)

        batch_size = min(5, len(self.ram))
        # iterate over last 10 random steps from pass game states and send them to our NN for learning.
        minibatch = random.sample(self.ram, batch_size)
        for state, action, reward, next_state, in minibatch:
            q_max = []
            # get MAX Q'(s', a') like in Q learning algo
            for action_x in bp.Policy.ACTIONS:
                q_max.append(self.model.predict(self.get_features(next_state, action_x))[0])
            # our target to fit our network - we get an vecotr of states and and action (Q,A) and we the network for it.
            target = reward + self.discount * np.max(q_max)
            s_features = self.get_features(state, action)
            target_f = self.model.predict(s_features)
            target_f[0] = target
            self.model.fit(s_features, target_f, epochs=1, verbose=0)
        self.memory = []
        if round % 10000 == 0:
            self.ram = []
        try:
            # use our epsilon decay method
            if round < self.game_duration - self.score_scope:
                self.epsilon -= self.decay_epsilon
           # if round % 100 == 0:
            if round > self.game_duration - self.score_scope:
                self.epsilon = 0
              #      self.log('round is: ' + str(round) + ' ' +"Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
              # else:
               #     self.log('round is: ' + str(round) + ' ' +"Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
               #self.r_sum = 0
            #else:
             #   self.r_sum += reward
        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > 1:
            self.remember(prev_state, prev_action, reward, new_state)
        if np.random.rand() < self.epsilon or round == 1:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            act_values = []
            #   for each action check the best action to make
            for action_x in bp.Policy.ACTIONS:
                act_values.append(self.model.predict(self.get_features(new_state, action_x))[0])
            b = act_values[::-1]
			# get the best action with highest score
            i = len(b) - np.argmax(b) - 1
            return bp.Policy.ACTIONS[i]


