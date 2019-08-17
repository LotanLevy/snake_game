

import numpy as np
from policies import Policy as bp

OBJ_RANGE = np.arange(-1, 10)
class Feature2:


    @staticmethod
    def obj_in_pos(pos, direction, obj_num, board):
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






    @staticmethod
    def features(new_state, action):
        """
        Creates feature vector according to the given state and action
        :param new_state:
        :param action:
        :return: feature vector
        """
        features_for_pos = np.zeros(11 * 2)

        board, head = new_state
        head_pos, direction = head

        next_direction = bp.Policy.TURNS[direction][action]
        next_pos = head_pos.move(next_direction)

        for i in range(11):
            features_for_pos[i] = int(board[next_pos[0], next_pos[1]] == OBJ_RANGE[i])
            features_for_pos[i + 11] = Feature2.obj_in_pos(next_pos, next_direction, OBJ_RANGE[i],
                                                       board)

        return features_for_pos.reshape((1, 22))