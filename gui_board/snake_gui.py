from tkinter import*
from random import choice
import numpy as np
from gui_board.Shape import Obstacle
from Constants import *
import scipy


from gui_board.Snake import Snake, KeyBoardSnake
from gui_board.BoardConstants import *
import threading


class Master(Canvas):
    """create the game canvas, the snake, the obstacle, keep track of the score"""
    def __init__(self, boss, random_food_prob, max_item_density, food_ratio, fruits_types={1:(1, OB_COLOR)}, start_func = None):
        super().__init__(boss)
        self.boss = boss
        self.width, self.height = WD, HT
        self.configure(width=self.width, height=self.height, bg=BG_COLOR)
        self.running = 0
        self.snakes = dict()
        self.fruits = dict()
        self.direction = None
        self.current = None
        self.fruits_types = fruits_types
        self.start_func = start_func
        self.func_thread = None

        self.board_array = self.init_board_array()

        self.score_manager = Score_manager(self.boss)
        self.fruit_id = 1
        self.item_count = 0

        self.random_food_prob = random_food_prob
        self.max_item_density = max_item_density
        self.food_ratio = food_ratio

    def join(self):
        if self.func_thread is not None:
            self.func_thread.join()



    def init_board_array(self):
        return np.ones((int(self.height/(2*PIXEL)), int(self.width/(2*PIXEL)))) * EMPTY_BOARD_FIELD

    def start(self):
        """start snake game"""
        if self.running == 0:
            for snake_id in self.snakes:
                self.snakes[snake_id].start()
            self.fruit_id += 1
            self.running = 1

            if self.start_func is not None:
                self.func_thread = threading.Thread(target=self.start_func)
                self.func_thread.start()




    def clean(self):
        """restarting the game"""
        if self.running == 1:
            self.running = 0
            for fruit in self.fruits.values():
                fruit.delete(from_board_dict=False)
            for snake in self.snakes.values():
                snake.delete()
            self.fruits = {}
            self.board_array = self.init_board_array()


    def add_sneak(self, sneak):
        self.score_manager.add_counter(sneak.id)
        self.snakes[sneak.id] = sneak
        sneak.update_score(self.score_manager)
        self.fruit_id = max(self.fruit_id, sneak.id)

    def update_array(self, x, y, id):
        self.board_array[int(((y /PIXEL) - 1)/2)][int(((x / PIXEL)-1)/2)] = id


    def get_position_in_board(self, array_coord_x, array_coord_y):
        return PIXEL * (2 * array_coord_y + 1), PIXEL * (2 * array_coord_x + 1)

    def get_position_in_array(self, x, y):
        return int(((y / PIXEL) - 1) / 2), int(((x / PIXEL) - 1) / 2)

    def get_board_cell_value(self, x, y):
        return self.board_array[int(((y /PIXEL) - 1)/2)][int(((x / PIXEL)-1)/2)]




    def add_fruit(self, a, b, type_id):
        self.fruit_id += 1
        new_fruit = Obstacle(self, self.fruit_id, self.fruits_types[type_id][1], type_id, a, b)
        self.fruits[self.fruit_id] = new_fruit

    def get_snake(self, id):
        return self.snakes[id]



    def get_board_array(self):
        board_with_fruit_types = np.copy(self.board_array)
        for id in self.fruits:
            x, y = np.where(self.board_array == id)
            board_with_fruit_types[x, y] = self.fruits[id].type_id
        return board_with_fruit_types

    def get_sneak_color(self, id):
        return self.snakes[id].color

    def delete_snake(self, id):
        self.snakes[id].delete_from_board()

    def move_sneak(self, id, direction, growing):
        self.snakes[id].change_direction(direction, growing)

    def update_score(self, id, new_value, r):
        self.score_manager.update_score(id, new_value, r)

    def move_step(self, orig_dir, x, y):
        new_dir = DIRECTIONS[MAIN_DIRECTION_MAP[orig_dir]]
        rows, cols = self.board_array.shape
        return (x + new_dir[1]) % rows,  (y + new_dir[0]) % cols

    def get_empty_slot(self, shape=(1,3)):
        is_empty = np.asarray(self.get_board_array() == EMPTY_CELL_VAL, dtype=int)
        match = scipy.signal.convolve2d(is_empty, np.ones(shape), mode='same') == np.prod(shape)
        if not np.any(match): raise ValueError('no empty slots of requested shape')
        r = np.random.choice(np.nonzero(np.any(match,axis=1))[0])
        c = np.random.choice(np.nonzero(match[r,:])[0])
        return (r,c)

    def randomize(self):
        if np.random.rand(1) < self.random_food_prob:
            if self.item_count < self.max_item_density * np.prod(self.board_array.shape):
                randfood = np.random.choice(list(FOOD_GROWING_MAP.keys()), 1)
                pos = self.get_empty_slot((1, 1))
                self.add_fruit(pos[0], pos[1], randfood[0])
                # self.item_count += 1
        return self.item_count

    def reset_player(self, id):

        positions = np.array(np.where(self.get_board_array() == id))
        self.delete_snake(id)
        # add fruits instead of parts of the snake location
        food_n = np.random.binomial(positions.shape[1], self.food_ratio)
        if self.item_count + food_n < self.max_item_density * np.prod(self.board_array.shape):
            subidx = np.array(np.random.choice(positions.shape[1], size=food_n, replace=False))
            if len(subidx) > 0:
                randfood = np.random.choice(list(FOOD_GROWING_MAP.keys()), food_n)
                for i,idx in enumerate(subidx):
                    self.add_fruit(positions[0,idx], positions[1,idx], randfood[i])
                # self.item_count += food_n






class Score_manager:

    def __init__(self, boss):
        self.maximum = StringVar(boss, '0')
        self.scores = {}
        self.boss = boss
        self.scoreboard = Frame(self.boss, width=35, height=2 * HT / 5)
        Label(self.scoreboard, text='High Score').grid()
        Label(self.scoreboard, textvariable=self.maximum).grid()

    def add_counter(self, sneak_id):
        if sneak_id not in self.scores:
            self.scores[sneak_id] = Score_Counter(self.boss, sneak_id, self.scoreboard)

    def update_score(self, sneak_id, new_value, r):
        self.scores[sneak_id].set(new_value, r)
        maximum = max(float(self.scores[sneak_id].counter.get()) , float(self.maximum.get()))
        self.maximum.set(str(maximum))

    def init_score_manager(self):
        self.scoreboard.grid(column=0, row=2)

    def delete_score(self, sneak_id):
        self.scores[sneak_id].delete()


class Score_Counter:

    def __init__(self, boss, id, scoreboard):
        self.counter = StringVar(boss, '0')
        self.id = id
        self.round = StringVar(boss, 'Mean score of sneak ' +str(id) + " in round " + str(0))
        self.title = Label(scoreboard, textvariable=self.round)
        self.title.grid()
        self.score = Label(scoreboard, textvariable=self.counter)
        self.score.grid()



    def set(self, new_value, r):
        score = new_value
        self.counter.set(str(score))
        self.round.set('Mean score of sneak ' +str(self.id) + " in round " + str(r))


    def delete(self):
        self.title.destroy()
        self.score.destroy()







class Scores:
    """Objects that keep track of the score and high score"""
    def __init__(self, boss):
        self.counter = StringVar(boss, '0')
        self.maximum = StringVar(boss, '0')

    def increment(self, score_addition):
        score = int(self.counter.get()) + score_addition
        maximum = max(score, int(self.maximum.get()))
        self.counter.set(str(score))
        self.maximum.set(str(maximum))

    def reset(self):
        self.counter.set('0')



def init_board_gui(fruits_types, start_func, random_food_prob, max_item_density, food_ratio):
    root = Tk()
    root.title("Snake Game")
    game = Master(root, random_food_prob, max_item_density, food_ratio, fruits_types=fruits_types, start_func=start_func)
    game.grid(column=1, row=0, rowspan=3)

    buttons = Frame(root, width=35, height=3*HT/5)
    Button(buttons, text='Start', command=game.start).grid()
    Button(buttons, text='Stop', command=game.clean).grid()
    Button(buttons, text='Quit', command=root.destroy).grid()
    buttons.grid(column=0, row=0)
    game.score_manager.init_score_manager()
    return root, game

def start_board(root, game):
    root.mainloop()
    game.join()
