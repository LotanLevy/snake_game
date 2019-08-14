from policies import Policy as bp
import numpy as np
from gui_board.snake_gui import Master
# from Snake_Main import Position
import Constants

EPSILON = 0.001
DISCOUNT_RATE = 0.1
LEARNING_RATE = 0.1
MY_INDEX = 0
ADVERSARY_INDEX = 1
#
# FEATURES_NUM = 22
#
# OBJ_RANGE = np.arange(-1, 10)

class Position():



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
        DIRECTIONS = {"N": [0, -1], 'S': [0, 1], 'E': [1, 0], "W": [-1, 0]}
        step = (DIRECTIONS[dir][1], DIRECTIONS[dir][1])
        return self + step

    def __copy__(self):
        position = Position((0, 0), board_size=self.board_size)
        position.pos = self.pos
        position.board_size = self.board_size
        return position

        # DIRECTIONS = {"N": [0, -1], 'S': [0, 1], 'E': [1, 0], "W": [-1, 0]}

        # if dir == 'E': return self + (0,1)
        # if dir == 'W': return self + (0,-1)
        # if dir == 'N': return self + (-1, 0)
        # if dir == 'S': return self + (1, 0)
        # raise ValueError('unrecognized direction')


class Direction():
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5

class Minmax(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes according to linear function we
    learned in class
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def __init__(self):
        super().__init__()

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
        indices_dict = dict()
        indices_dict[MY_INDEX] = self.id
        indices_dict[ADVERSARY_INDEX] = self.adversaries_ids[0]
        return self.minmax(new_state, 4, MY_INDEX, indices_dict)[1]

    def evaluation_function(self, state, indices_dict):
        # returns the hamilton distance between the two snakes heads:
        snake_1_head = state[1][0]     # gets the head of snake_1
        snake_2_head = self.get_snake_head_position(state, ADVERSARY_INDEX, indices_dict)
        dist = abs(snake_1_head[0] - snake_2_head[0]) + abs(snake_1_head[1] - snake_2_head[1])
        return dist

    def get_some_body_part(self, state, snake_id):
        """
        returns some element from the snake's body parts
        """
        return np.where(state == snake_id)

    def minmax(self, state, depth, player_index, indices_dict): #TODO: understand if it starts with 0 or 1. how to connect snake id to player index?

        if depth == 0:
            return self.evaluation_function(state), Direction.STOP

        available_directions = self.get_legal_actions(state)
        successor_states = [self.generate_successor(player_index, state, direction, indices_dict) for direction in available_directions]
        scores = []
        for successor in successor_states:
            result = self.minmax(successor, depth - 1, 1 - player_index)
            scores.append(result[0])

        if player_index == MY_INDEX:
            best_score_idx = np.argmax(np.array(scores))
        else:
            best_score_idx = np.argmin(np.array(scores))

        return scores[best_score_idx], available_directions[best_score_idx]

    def generate_successor(self, player_index, state, indices_dict, direction=Direction.STOP):
        """
        creates a copy of the given game object, moves the current players' snake (0 or 1) by the given action
        :return: successor's game (after performing the action)
        """
        # copy the game board
        copied_state = self.copy_state(state)
        # do move - requires direction of the snake, snake's head and the board's borders
        # I used snake_gui.move_step because it will be a static method

        board = self.apply_step_to_successor(copied_state[0], player_index, indices_dict, direction)
        new_state = board, copied_state[1]
        return new_state

    def get_snake_head_position(self, state, player_index, indices_dict):
        current_board = state[0]
        copied_state = self.copy_state(state)
        board = self.apply_step_to_successor(copied_state[0], player_index, indices_dict, "F")
        new_coords = np.where(board == indices_dict[player_index])
        current_coords = np.where(current_board == indices_dict[player_index])
        head_position = (0, 0)
        for (x, y) in new_coords:
            if (x, y) not in current_coords:
                head_position = (x, y)
        return head_position

    def apply_step_to_successor(self, board, player_index, indices_dict, direction):
        moved_head_coords = Master.move_step(direction, board, self.board_size)
        board[moved_head_coords] = indices_dict[player_index]
        return board

    def copy_state(self, state):
        new_position = Position.__copy__(state[1][0])
        head = (new_position, state[1][1])
        new_state = state[0], head
        return new_state

    def get_legal_actions(self, state):
        direction = state[1][0]
        legal_directions = [Direction.LEFT, Direction.RIGHT, Direction.DOWN, Direction.UP]
        if direction == Direction.LEFT:
            legal_directions.remove(Direction.RIGHT)
        elif direction == Direction.RIGHT:
            legal_directions.remove(Direction.LEFT)
        elif direction == Direction.UP:
            legal_directions.remove(Direction.DOWN)
        else:
            legal_directions.remove(Direction.UP)
        return legal_directions


    # #TODO: implement this func
    # def apply_action(self, action):
    #     pass
    #
    #     # move our snake according to the action
    #     ####TODO: we need an access from the state (Game object) to the main player's snake,
    #         # and to the other player's snake (and to know who is who)...
    #
    # # TODO: implement this func
    # def apply_opponent_action(action):
    #     pass
    #     # move opponent snake according to the action
    #
