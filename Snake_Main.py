import datetime
import multiprocessing as mp
import queue
import gzip
from gui_board.BoardConstants import *
from Constants import *
from Agent import Agent

from config import parse_args

import numpy as np
import scipy.signal as ss

from policies import Policy

from gui_board.snake_gui import init_board_gui, start_board
from gui_board.Snake import Snake
import time
from save_data import csvWriter


def days_hours_minutes_seconds(td):
    """
    parse time for logging.
    """
    return td.days, td.seconds//3600, (td.seconds//60)%60, td.seconds%60


def random_partition(num, max_part_size):
    parts = []
    while num > 0:
        parts.append(np.random.randint(1, min(max_part_size, num+1)))
        num -= parts[-1]
    return parts


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
        step = (DIRECTIONS[dir][1], DIRECTIONS[dir][1])
        return self + step

        # DIRECTIONS = {"N": [0, -1], 'S': [0, 1], 'E': [1, 0], "W": [-1, 0]}

        # if dir == 'E': return self + (0,1)
        # if dir == 'W': return self + (0,-1)
        # if dir == 'N': return self + (-1, 0)
        # if dir == 'S': return self + (1, 0)
        # raise ValueError('unrecognized direction')




class Game(object):

    @staticmethod
    def log(q, file_name, on_screen=True):
        start_time = datetime.datetime.now()
        logfile = None
        if file_name:
            logfile = gzip.GzipFile(file_name,
                                    'w') if file_name.endswith(
                '.gz') else open(file_name, 'wb')
        for frm, type, msg in iter(q.get, None):
            td = datetime.datetime.now() - start_time
            msg = '%i::%i:%i:%i\t%s\t%s\t%s' % (
            days_hours_minutes_seconds(td) + (frm, type, msg))
            if logfile: logfile.write((msg + '\n').encode('ascii'))
            if on_screen: print(msg)
        if logfile: logfile.close()



    def __init__(self, args_and_names):
        args, fields_name = args_and_names

        self.__dict__.update(args.__dict__)

        self.csv = csvWriter(["round"] + fields_name, "csv_game_eps_f=2_e=0.001_dr.csv")

        # check that the number of players is OK:
        self.n = len(self.policies)
        assert self.n <= MAX_PLAYERS, "Too Many Players!"

        self.round = 0

        # init gui

        self.board_root, self.game_gui = init_board_gui(FOOD_COLOR_MAP, self.run, self.random_food_prob, self.max_item_density, self.food_ratio) #gui

        # init logger
        self.logq = mp.Queue()
        # to_screen = not self.to_render
        self.logger = mp.Process(target=self.log, args=(self.logq, self.log_file))
        self.logger.start()

        # initialize the board:
        # self.board = EMPTY_CELL_VAL * np.ones(self.board_size, dtype=int)
        self.previous_board = None
        self.init_player_size = args.init_player_size

        # initialize players:
        self.rewards, self.players, self.scores, self.directions, self.actions, self.growing, self.size, self.previous_heads = \
            [], [], [], [], [], [], [], []

        snakes_ids = range(len(self.policies))

        for i, (policy, pargs) in enumerate(self.policies):
            color = pargs['color'] if 'color' in pargs else SN_COLOR
            self.init_player(i, color)


            self.rewards.append(0)
            self.actions.append(None)
            self.previous_heads.append(None)
            self.scores.append([0])
            adversaries = list(snakes_ids)
            adversaries.remove(i)
            self.players.append(Agent(i, policy, pargs, self.board_size, self.logq, self.game_duration, self.score_scope, adversaries, self.game_gui.get_fruit_types()))
        # wait for player initialization (Keras loading time):
        time.sleep(self.player_init_time)


    def init_player(self, id, color, reset=False):

        # initialize the position and direction of the player:
        dir = np.random.choice(list(Policy.Policy.TURNS.keys()))
        shape = (1, 3) if dir in ['W', 'E'] else (3, 1)
        first = self.game_gui.get_empty_slot(shape)
        sec = self.game_gui.move_step(dir, first[0], first[1], self.game_gui.board_array.shape)
        assert self.game_gui.get_board_array()[sec[0], sec[1]] == EMPTY_CELL_VAL

        snake_gui = Snake(self.game_gui, id,  color, self.init_player_size - 2)
        positions = [first, sec]

        snake_gui.set_positions(positions)
        snake_gui.set_direction(dir)
        self.game_gui.add_sneak(snake_gui)
        if reset:
            snake_gui.start()


    def play_a_round(self):

        # randomize the players:
        pperm = np.random.permutation([(i,p) for i, p in enumerate(self.players)])

        # distribute states and rewards on previous round
        board = self.game_gui.get_board_array()
        for i, p in pperm:
            snake = self.game_gui.get_snake(p.id)
            x, y = snake.get_head()
            current_head = (Position((x,y), self.board_size), snake.direction)
            if self.previous_board is None:
                p.handle_state(self.round, None, self.actions[p.id], self.rewards[p.id], (board, current_head))
            else:
                p.handle_state(self.round, (self.previous_board, self.previous_heads[p.id]), self.actions[p.id], self.rewards[p.id], (board, current_head))
            self.previous_heads[p.id] = current_head
        self.previous_board = np.copy(board)

        # wait and collect actions
        time.sleep(self.policy_action_time)
        actions = {p: p.get_action() for _, p in pperm}
        if self.round % LEARNING_FREQ == 0 and self.round > 5:
            time.sleep(self.policy_learn_time)

        # get the interactions of the players with the board:
        for _, p in pperm:
            snake = self.game_gui.get_snake(p.id)
            action = actions[p]
            self.actions[p.id] = action
            head = snake.get_head()
            move_to = self.game_gui.move_step(Policy.Policy.TURNS[snake.direction][action], head[0], head[1], self.game_gui.board_array.shape)
            # reset the player if he died:
            if board[move_to[0], move_to[1]] != EMPTY_CELL_VAL and board[move_to[0], move_to[1]] not in FOOD_GROWING_MAP:
                self.game_gui.reset_player(p.id)
                self.init_player(p.id, self.game_gui.get_sneak_color(p.id), True)
                # self.size[p.id] = player_size
                # self.growing[p.id] = growing
                # self.directions[p.id] = dir
                self.rewards[p.id] = DEATH_PENALTY
                self.scores[p.id].append(self.rewards[p.id])


            # otherwise, move the player on the board:
            else:
                self.rewards[p.id] = 0
                if board[move_to[0], move_to[1]] in FOOD_GROWING_MAP.keys():
                    self.rewards[p.id] += FOOD_SCORE_MAP[board[move_to[0], move_to[1]]]
                    self.game_gui.get_snake(p.id).growing += FOOD_GROWING_MAP[board[move_to[0], move_to[1]]]  # start growing

                self.game_gui.move_sneak(p.id, action)
                self.scores[p.id].append(self.rewards[p.id])

        # update the food on the board:
        self.game_gui.randomize()
        self.round += 1

    def run(self):
        try:
            r = 0
            while r < self.game_duration:
                r += 1
                time.sleep(0.2)
                if r % STATUS_UPDATE == 0:
                    print("At Round " + str(r) + " the scores are:")
                    self.csv.build_row(0,r)

                    for i, s in enumerate(self.scores):
                        scope = np.min([len(self.scores[i]), self.score_scope])
                        print("Player " + str(i + 1) + ": " + str(str("{0:.4f}".format(np.mean(self.scores[i][-scope:])))))
                        self.csv.build_row(i + 1, str(
                            "{0:.4f}".format(np.mean(self.scores[i][-scope:]))))
                        self.game_gui.update_score(i, np.mean(self.scores[i][-scope:]), r)



                self.play_a_round()

        finally:
            output = [','.join(['game_id','player_id','policy','score'])]
            game_id = str(abs(id(self)))
            for p, s in zip(self.players, self.scores):
                p.shutdown()
                pstr = str(p.policy).split('<')[1].split('(')[0]
                scope = np.min([len(s), self.score_scope])
                p_score = np.mean(s[-scope:])
                oi = [game_id, str(p.id), pstr, str("{0:.4f}".format(p_score))]
                output.append(','.join(oi))

            with open(self.output_file, 'w') as outfile:
                outfile.write('\n'.join(output))
            self.logq.put(None)
            self.logger.join()


if __name__ == '__main__':
    g = Game(parse_args())
    start_board(g.board_root, g.game_gui)
    g.csv.create_csv()
