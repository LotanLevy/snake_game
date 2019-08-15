from policies import Policy as bp
import numpy as np
from gui_board.snake_gui import Master

EPSILON = 0.001
DISCOUNT_RATE = 0.1
LEARNING_RATE = 0.1
OUR_AGENT = 1
OPPONENT_AGENT = 0


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

    def __init__(self):
        super().__init__()

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

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
        return self.minmax(new_state, 4, self.id, self.fruits_ids)[1]

    def evaluation_function(self, state, fruits_ids):

        #TODO evaluate states by the snake's closeness to the fruit. the closer + positive value = better
        distances = []
        snake_head = state[0] # Position element
        for fruit_id in fruits_ids:
            positions_tuple = np.where(state[0] == fruit_id)

        for i in range(len(positions_tuple[0])):
            distances.append(abs(snake_head[0] - positions_tuple[0][0]) + abs(snake_head[1] - positions_tuple[1][0]))

        return dist

    def minmax(self, state, depth, player_index, fruits_ids):

        if depth == 0:
            return self.evaluation_function(state, fruits_ids), Direction.STOP

        available_directions = self.get_legal_actions(state)
        successor_states = [self.generate_successor(player_index, direction) for direction in available_directions]
        scores = []
        for successor in successor_states:
            result = self.minmax(successor, depth - 1, 1 - player_index)
            scores.append(result[0])

        if player_index == OUR_AGENT:
            best_score_idx = np.argmax(np.array(scores))
        else:
            best_score_idx = np.argmin(np.array(scores))

        return scores[best_score_idx], available_directions[best_score_idx]

    def generate_successor(self, state, direction=Direction.STOP):
        """
        creates a copy of the given game object, moves the current players' snake (0 or 1) by the given action
        :return: successor's game (after performing the action)
        """
        # copy the game board
        copied_state= np.copy(state)
        # do move - requires direction of the snake, snake's head and the board's borders
        # I used snake_gui.move_step because it will be a static method
        successor_state = Master.move_step(direction, state[1], (state.board_shape[0], state.board_shape[1]))
        return copied_state

    def get_legal_actions(self, state):
        direction = state[1].direction
        legal_actions = [Direction.LEFT, Direction.RIGHT, Direction.DOWN, Direction.UP]
        if direction == Direction.LEFT:
            legal_actions.remove(Direction.RIGHT)
        elif direction == Direction.RIGHT:
            legal_actions.remove(Direction.LEFT)
        elif direction == Direction.UP:
            legal_actions.remove(Direction.DOWN)
        else:
            legal_actions.remove(Direction.UP)
        return legal_actions


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
