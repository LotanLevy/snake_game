from tkinter import*
from random import choice
import numpy as np
from gui.GuiComponents import BoardSnake, Fruit, BoardScore, Obstacle
from GameComponants import Position
from GameConfigurations import *
from scipy import signal
import threading

import time


def init_board_array():
    return np.ones((int(WD / (2 * PIXEL)), int(HT / (2 * PIXEL)))) * EMPTY_BOARD_FIELD


class Board(Canvas):
    """create the game canvas, the snake, the obstacle, keep track of the score"""
    def __init__(self, tk_root, board_width, board_height, fruits_types, random_food_prob, max_item_density, obstacle_density, start_func, snakes):
        super().__init__(tk_root)
        self.tk_root = tk_root
        self.board_array = np.ones((board_width, board_height)) * EMPTY_BOARD_FIELD
        width, height = self.get_gui_board_dim()
        self.gui_board_shape = (2 * height, 2* width)
        self.configure(width=2 * height, height= 2* width, bg=BG_COLOR)
        self.random_food_prob = random_food_prob
        self.max_item_density = max_item_density
        self.start_func = start_func
        self.obstacle_density = obstacle_density


        self.fruits_types = fruits_types
        self.board_items = 0

        # Gui items
        self.snakes = dict()
        self.fruits = dict()
        self.obstacles = []
        self.scoreboard = Frame(self.tk_root, width=35, height=2 * height / 5)
        self.scoreboard.grid(column=0, row=2)

        self.fruit_id = 1
        self.running = 0

        for snake in snakes.values():
            self.initiate_snake(snake)

    @staticmethod
    def random_partition(num, max_part_size):
        parts = []
        while num > 0:
            parts.append(np.random.randint(1, min(max_part_size, num + 1)))
            num -= parts[-1]
        return parts

    def add_obstacles(self):
        # initialize obstacles:
        obstacle_num = int(self.obstacle_density * np.prod(self.board_array.shape))
        for size in self.random_partition(obstacle_num, np.min(self.board_array.shape)):
            r = np.random.rand(1)
            shape = (size, 1) if r > 0.5 else (1, size) # obstacle direction
            positions = []
            pos = self.get_empty_slot(self.board_array, shape=shape)
            for i in range(size):
                step = (i, 0) if r > 0.5 else (0, i) # obstacle direction
                new_pos = pos + step
                self.board_array[new_pos[0], new_pos[1]] = OBSTACLE_VAL
                positions.append(new_pos)
            self.obstacles.append(Obstacle(self, OBSTACLE_VAL, positions, OB_COLOR))
            self.board_items += size



    def start(self):
        if self.running == 0:
            for id in self.snakes:
                self.snakes[id]["snake_gui"].start()
            self.add_obstacles()
            for o in self.obstacles:
                o.start()
            self.add_food()
            self.running = 1
            if self.start_func is not None:
                self.func_thread = threading.Thread(target=self.start_func)
                self.func_thread.start()

    @staticmethod
    def get_empty_slot(board_array, shape=(1,3)):
        is_empty = np.asarray(board_array == EMPTY_BOARD_FIELD, dtype=int)
        match = signal.convolve2d(is_empty, np.ones(shape), mode='same') == np.prod(shape)
        if not np.any(match): raise ValueError('no empty slots of requested shape')
        r = np.random.choice(np.nonzero(np.any(match,axis=1))[0])
        c = np.random.choice(np.nonzero(match[r,:])[0])
        return Position((r,c), board_array.shape)

    def get_gui_board_dim(self): # returns width, height
        shape = self.board_array.shape
        return shape[0] * PIXEL, shape[1] * PIXEL


    def add_food(self):
        if np.random.rand(1)[0] < self.random_food_prob:
            self.board_items = len(self.fruits.keys())
            if self.board_items < self.max_item_density * np.prod(self.board_array.shape):
                randfood = np.random.choice(list(FOOD_MAP.keys()), 1)[0]
                slot = self.get_empty_slot(self.board_array, (1, 1))
                self.board_array[slot[0], slot[1]] = randfood
                self.fruits[self.food_repre_by_pos(slot)] = Fruit(self, randfood, slot[0], slot[1], self.fruits_types[randfood]["color"], randfood)
                self.board_items += 1


    def food_repre_by_pos(self, pos):
        return str(pos[0]) + " " + str(pos[1])


    def initiate_snake(self, snake):
        snake.direction = np.random.choice(list(DIRECTIONS.keys()))
        shape = (1, 3) if snake.direction in ['W', 'E'] else (3, 1)
        snake.size = len(shape)
        snake.growing = snake.init_player_size - snake.size

        # snake positions #
        first = Board.get_empty_slot(self.board_array, shape)
        sec = first.move(snake.direction)

        # update snake object #
        snake.positions = [first, sec]

        # update board_array #
        for pos in snake.positions:
            self.board_array[pos[0], pos[1]] = snake.id

        # update the gui board
        new_board_snake = BoardSnake(self, snake.id, snake.color, "white", snake.positions, self.gui_board_shape)
        if snake.id in self.snakes:
            self.snakes[snake.id]["snake_gui"].delete()
            self.snakes[snake.id]["snake_gui"] = new_board_snake
        else:
            self.snakes[snake.id] = {"snake_obj": snake,
                                     "snake_gui": new_board_snake,
                                     "snake_score": BoardScore(self.tk_root, self.scoreboard,
                                                               snake.get_gui_representation())}
        self.board_items += len(snake.positions)


    def restart_snake(self, snake):
        self.board_array[self.board_array == snake.id] = EMPTY_BOARD_FIELD
        self.board_items -= len(snake.positions)
        self.initiate_snake(snake)
        self.snakes[snake.id]["snake_gui"].start()

    def handle_collision(self, snake):
        snake.update_score(THE_DEATH_PENALTY)
        Board.restart_snake(self, snake)

    def move_gui_board_snake(self, snake, can_grow):
        self.snakes[snake.id]["snake_gui"].move(snake.direction, can_grow)

    def delete_food_from_board(self, food_pos):
        self.fruits[self.food_repre_by_pos(food_pos)].delete()
        del self.fruits[self.food_repre_by_pos(food_pos)]

    @staticmethod
    def move_snake(snake, direction, board_array, snakes_ids, food_types):
        """
        :param snake: snake object
        :param direction: The snake's new direction movement
        :param board_array: the board array to update
        :param snakes_ids: all the snakes ids in the board
        :param obstacles_ids: all the obstacles ids in the board
        :param food_types: a dictionary represents the game's food types
        :return: "collision" if the snake died and "food" if the snake ate a fruit and "" otherwise
        if the snake died the user should call handle_collision afterward. This function will restart the snake in the gui too
        In order to update the board with a movement that succeeded the user should call move_gui_board_snake
        If the snake ate food you should call delete_food_from_snake in order to delete the food from the gui board
        """
        head = snake.positions[-1]
        new_pos = head.move(direction)
        board_value = board_array[new_pos[0], new_pos[1]]
        result = ""
        will_grow = False
        if board_value in snakes_ids or board_value == OBSTACLE_VAL: # collision
            result = "collision"
        else:
            if board_value in food_types.keys():         #ate food
                snake.current_score += food_types[board_value]["score"]
                snake.growing += food_types[board_value]["growing"]
                board_array[new_pos[0], new_pos[1]] = EMPTY_BOARD_FIELD
                result = "food"
            snake.direction = direction
            will_grow = snake.move(new_pos, board_array)
        return new_pos, result, board_value, will_grow

    def update_scores(self, round, score_scope):

        # max_score = self.max_score.get_score()

        for id in self.snakes:
            snake = self.snakes[id]["snake_obj"]
            scope = np.min([len(snake.all_scores), score_scope])
            snake_score = np.mean(snake.all_scores[-scope:])
            self.snakes[id]["snake_score"].update(snake_score, round)


    def join(self):
        if self.func_thread is not None:
            self.func_thread.join()













def init_board_gui(height, width, fruits_types, random_food_prob, max_item_density, obstacle_density, start_func, snakes):
    root = Tk()
    root.title("Snake Game")
    game = Board(root, width, height, fruits_types, random_food_prob, max_item_density, obstacle_density, start_func, snakes)
    game.grid(column=1, row=0, rowspan=3)

    buttons = Frame(root, width=35, height=3*HT/5)
    Button(buttons, text='Start', command=game.start).grid()
    Button(buttons, text='Quit', command=root.destroy).grid()
    buttons.grid(column=0, row=0)
    return root, game

def start_board(root, game):
    root.mainloop()
    game.join()


