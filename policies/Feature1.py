import numpy as np
from policies import Policy as bp


class Feature1:

    ## LOTAN - START

    # @staticmethod
    # def is_board_edge(board, x, y, direction):
    #     """
    #     Checks if we are on board edge
    #     :param board: board
    #     :param x: vertical position
    #     :param y: horizontal position
    #     :param direction: direction
    #     :return: true/false if we are on board edge
    #     """
    #     if direction == 'N' and x == 0:
    #         return True
    #     if direction == 'S' and x == board.shape[0]-1:
    #         return True
    #     if direction == 'E' and y == board.shape[1]-1:
    #         return True
    #     if direction == 'W' and y == 0:
    #         return True
    #     return False
    #
    # @staticmethod
    # def values_in_direction(state, direction, max_steps):
    #     """
    #     Checks if in the given direction there is a good-fruit
    #     :param state: state
    #     :param direction: direction
    #     :return: true/false and the position, if true
    #     """
    #     board, head = state
    #     cur_pos, head_direction = head
    #     direct_vec = np.zeros((11,))
    #     steps = 0
    #
    #     while (not Feature1.is_board_edge(board, cur_pos[0], cur_pos[1], direction)) and \
    #             max_steps >= steps:
    #         cur_pos = cur_pos.move(direction)
    #         board_value = board[cur_pos[0], cur_pos[1]]
    #         direct_vec[board_value] = 1
    #         steps += 1
    #     return direct_vec.tolist()
    #
    # @staticmethod
    # def directionFeature(direction):
    #     """ return a feature that represent the direction """
    #     direction_feature = []
    #     direction_feature.append(int(direction == 'N'))
    #     direction_feature.append(int(direction == 'S'))
    #     direction_feature.append(int(direction == 'W'))
    #     direction_feature.append(int(direction == 'E'))
    #     return direction_feature
    #
    #
    # @staticmethod
    # def actionFeature(action):
    #     """ return a feature that represent the action """
    #     action_feature = []
    #     action_feature.append(int(action == 'L'))
    #     action_feature.append(int(action == 'R'))
    #     action_feature.append(int(action == 'F'))
    #     return action_feature
    #
    # @staticmethod
    # def value_in_pos(head_pos, board):
    #     """
    #     :param head_pos: head position
    #     :param board: board
    #     :return: zeros vector of size (11,) with 1 at the index of the value in the next position
    #     """
    #     values = np.zeros((11,))
    #     board_value = board[head_pos[0], head_pos[1]]
    #     values[board_value+1] = 1
    #     return values.tolist()
    #
    # @staticmethod
    # def features_around_pos(action, head, board):
    #     head_pos, direction = head
    #     next_direction = bp.Policy.TURNS[direction][action]
    #     next_pos = head_pos.move(next_direction)
    #
    #     values = np.zeros((11,))
    #     for next_action in bp.Policy.ACTIONS:
    #         next_next_direction = bp.Policy.TURNS[next_direction][next_action]
    #         next_next_pos = next_pos.move(next_next_direction)
    #         values[board[next_next_pos[0], next_next_pos[1]]+1] = 1
    #     return values.tolist()

    ## LOTAN - END

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
        A function that fill the features around the snake head position
        :param action: action
        :param head: head
        :param board: board
        :param level: steps to look ahead
        :return: list of the values around the snake head
        """
        head_pos, direction = head
        values = np.zeros((11,))
        Feature1.features_around_pos_with_level_helper(action, head_pos, direction, board, level, values)
        return values.tolist()

    @staticmethod
    def features_around_pos_with_level_helper(action, cur_pos, cur_direction, board, max_level, values):
        """
        Helper function that fill the features around the snake head position
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
            values[board[next_pos[0], next_pos[1]]+1] = 1
            return

        for next_action in bp.Policy.ACTIONS:
            Feature1.features_around_pos_with_level_helper(next_action, next_pos, next_direction, board, new_level, values)

    @staticmethod
    def build_state_vec(state, level=4):
        """
        Creates feature vector according to the given state and action
        :param state:
        :param action:
        :return: feature vector
        """
        features_for_pos = []
        board, head = state
        head_pos, direction = head

        # 3 vector, vector for each direction - indicates the cell value after acting in that
        # direction (33 features)
        for action in bp.Policy.ACTIONS:
            features_for_pos += Feature1.next_action_feature(head_pos, direction,
                                                                         action, board)
            if level >= 2:
                features_for_pos += Feature1.features_around_pos_with_level(action, head, board, 2)
            if level >= 3:
                features_for_pos += Feature1.features_around_pos_with_level(action, head, board, 3)
            if level >= 4:
                features_for_pos += Feature1.features_around_pos_with_level(action, head, board, 4)

        return np.reshape(np.array(features_for_pos), (1,len(features_for_pos)))