from policies import Policy as bp
import numpy as np
from scipy.spatial import distance
import copy
from gui.gui_board import Board

EPSILON = 0.001
DISCOUNT_RATE = 0.1
LEARNING_RATE = 0.1
MY_INDEX = 0
ADVERSARY_INDEX = 1

ACTIONS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
}

#
# FEATURES_NUM = 22
#
# OBJ_RANGE = np.arange(-1, 10)


class Position:

    def __init__(self, position, board_size):
        self.pos = position
        self.board_size = board_size

    def __getitem__(self, key):
        return self.pos[key]

    def __add__(self, other):
        return Position(((self[0] + other[0]) % self.board_size[0],
                        (self[1] + other[1]) % self.board_size[1]),
                        self.board_size)

    def move(self, dir):

        # new_move = self.gui.move_step(dir, self.pos[0], self.pos[1])
        if dir == 'E': return self + (0,1)
        if dir == 'W': return self + (0,-1)
        if dir == 'N': return self + (-1, 0)
        if dir == 'S': return self + (1, 0)
        raise ValueError('unrecognized direction')

    def __copy__(self):
        position = Position((0, 0), board_size=self.board_size)
        position.pos = self.pos
        position.board_size = self.board_size
        return position


class Minmax(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes according to linear function we
    learned in class
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        print("initiating minmax")
        self.r_sum = 0
        for snake in self.snakes:
            if snake.id != self.id:
                self.adversary_id = snake.id
                break

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
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
        """
        The function for choosing an action, given current state.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the policy for a few rounds in a row.
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        # for now we use only one adversary
        print("min max \"act\" function is activated")
        indices_dict = dict()
        indices_dict[MY_INDEX] = self.id
        indices_dict[ADVERSARY_INDEX] = self.adversary_id

        return self.minmax(new_state, 4, MY_INDEX, indices_dict)[1]

    def evaluation_function(self, state, indices_dict):
        """
        by distance from the closest point to the current snake's head
        :param state:
        :param indices_dict:
        :return:
        """

        head0 = state.snakes[indices_dict[MY_INDEX]].positions[-1]
        head1 = state.snakes[indices_dict[ADVERSARY_INDEX]].positions[-1]

        # hamilton distance:
        dist = abs(head0[0] - head1[0]) + abs(head0[1] - head1[1])
        # TODO: add attention to food dist etc...
        return dist

    def minmax(self, state, depth, player_index, indices_dict):

        if depth == 0:
            return self.evaluation_function(state, indices_dict), 'F'

        possible_actions = ['L', 'R', 'F']
        successor_states = [self.generate_successor(player_index, state, indices_dict, action) for action in possible_actions]

        scores = []
        for successor in successor_states:
            result = self.minmax(successor, depth - 1, 1 - player_index, indices_dict)
            scores.append(result[0])

        if player_index == MY_INDEX:
            best_score_idx = np.argmax(np.array(scores))
        else:
            best_score_idx = np.argmin(np.array(scores))

        return scores[best_score_idx], possible_actions[best_score_idx]

    def generate_successor(self, player_index, state, indices_dict, action):
        """
        creates a copy of the given game object, moves the current players' snake (0 or 1) by the given action
        :return: successor's game (after performing the action)
        """
        # do move - requires direction of the snake, snake's head and the board's borders
        # I used snake_gui.move_step because it will be a static method

        direction = bp.Policy.TURNS[state[1][1]][action]

        board, new_snake, head_pos = self.apply_step_to_successor(state, player_index, indices_dict, direction)
        new_state = board, (head_pos, direction)
        return new_state

    def apply_step_to_successor(self, state, player_index, indices_dict, direction):
        # copy the game board
        copied_state = self.copy_state(state)
        copied_snake = copy.deepcopy(copied_state.snakes[indices_dict[player_index]])
        new_board, new_snake, head_pos = Board.move_snake(copied_snake, direction, copied_state,
                                                                      copied_state[0].fruits_types)
        return new_board, new_snake, head_pos

    def copy_state(self, state):
        new_position = Position.__copy__(state[1][0])
        head = (new_position, state[1][1])
        board = np.array(state[0], copy=True)
        new_state = board, head
        return new_state

    def get_legal_actions(self, state):
        direction = state[1][1]
        legal_directions = ['E', 'W', 'N', 'S']
        if direction == 'E':
            legal_directions = ['E', 'N', 'S']
        elif direction == 'W':
            legal_directions = ['W', 'N', 'S']
        elif direction == 'N':
            legal_directions = ['E', 'W', 'N']
        elif direction == 'S':
            legal_directions = ['E', 'W', 'S']
        return legal_directions
