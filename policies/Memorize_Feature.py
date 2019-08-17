


class Memorize_Feture:

    def update_rewards(self, state, reward):
        """
        Learn the board representation in order to learn the different objects rewards
        :param state: state
        :param reward: reward
        :return:
        """
        board, head = state
        head_pos, direction = head

        if self.last_board is not None:
            value = self.last_board[head_pos[0], head_pos[1]]

            if value not in self.rewards:
                self.rewards[value] = set()
            self.rewards[value].add(reward)
        self.last_board = board

    def is_obstacle(self, board_value, good_fruit = False):
        """
        Checks if the object in the given board_value is an obstacle
        :param board_value: the object board representation
        :param good_fruit: true in order to check if we are on a good fruit
        :return: true/false if we are on a obstacle
        """
        if board_value not in self.rewards or ZERO in self.rewards[board_value]:
            return False
        if good_fruit:
            return list(self.rewards[board_value])[0] > 0

        return list(self.rewards[board_value])[0] < 0

    def is_board_edge(self, board, x, y, direction):
        """
        Checks if we are on board edge
        :param board: board
        :param x: vertical position
        :param y: horizontal position
        :param direction: direction
        :return: true/false if we are on board edge
        """
        if direction == 'N' and x == 0:
            return True
        if direction == 'S' and x == board.shape[0]-1:
            return True
        if direction == 'E' and y == board.shape[1]-1:
            return True
        if direction == 'W' and y == 0:
            return True
        return False

    def has_fruit_in_direction(self, state, direction):
        """
        Checks if in the given direction there is a good-fruit
        :param state: state
        :param direction: direction
        :return: true/false and the position, if true
        """
        board, head = state
        cur_pos, head_direction = head

        while not self.is_board_edge(board, cur_pos[0], cur_pos[1], direction):
            cur_pos = cur_pos.move(direction)
            board_value = board[cur_pos[0], cur_pos[1]]
            if self.is_obstacle(board_value, True):
                return True, cur_pos
        return False, -1

    def dist(self, cur_pos, new_pos):
        """
        Calculates euclidean distance
        :param cur_pos: cur_pos
        :param new_pos: new_pos
        :return: euclidean distance
        """
        return np.sqrt(np.power(cur_pos[0] - new_pos[0], 2) + np.power(cur_pos[1] - new_pos[1], 2))





    def build_state_vec(self, new_state):
        """
        Creates feature vector according to the given state and action
        :param new_state:
        :param action:
        :return: feature vector
        """
        features_for_pos = []

        board, head = new_state
        head_pos, direction = head

        # 3 feature for danger for right, left and forward steps
        for action in bp.Policy.ACTIONS:
            next_direction = bp.Policy.TURNS[direction][action]
            next_pos = head_pos.move(next_direction)

            board_value = board[next_pos[0], next_pos[1]]

            features_for_pos.append(int(self.is_obstacle(board_value)))

        # 4 feature for directions
        features_for_pos.append(int(direction == 'N'))
        features_for_pos.append(int(direction == 'S'))
        features_for_pos.append(int(direction == 'W'))
        features_for_pos.append(int(direction == 'E'))

        # 4 feature for good fruit in specific direction
        directions = ['N', 'S', 'W', 'E']

        closest_pos = None
        dist = None
        best_direction = None

        for d in directions:
            has_fruit, pos = self.has_fruit_in_direction(new_state, d)
            if has_fruit:
                new_dist = self.dist(head_pos, pos)
                if closest_pos is None or self.dist(head_pos, pos) < dist:
                    closest_pos = pos
                    dist = new_dist
                    best_direction = d

        for d in directions:
            if best_direction is not None and best_direction == d:
                features_for_pos.append(1)
            else:
                features_for_pos.append(0)
        return np.reshape(np.array(features_for_pos), (1,11))