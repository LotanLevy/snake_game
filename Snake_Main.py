import datetime
import multiprocessing as mp
import queue
import gzip
from gui_board.BoardConstants import *
from Constants import *

from config import parse_args

import numpy as np
import scipy.signal as ss

from policies import Policy

from gui_board.snake_gui import init_board_gui, start_board
from gui_board.Snake import Snake

import time


from save_data import csvWriter








def clear_q(q):
    """
    given a queue, empty it.
    """
    while not q.empty():
        try: q.get_nowait()
        except queue.Empty: break


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

        # new_move = self.gui.move_step(dir, self.pos[0], self.pos[1])
        if dir == 'E': return self + (0,1)
        if dir == 'W': return self + (0,-1)
        if dir == 'N': return self + (-1, 0)
        if dir == 'S': return self + (1, 0)
        raise ValueError('unrecognized direction')




class Agent(object):
    SHUTDOWN_TIMEOUT = 60 # seconds until policy is considered unresponsive

    def __init__(self, id, policy, policy_args, board_size, logq, game_duration, score_scope):
        """
        Construct a new player
        :param id: the player id (the value of the player positions in the board
        :param policy: the class of the policy to be used by the player
        :param policy_args: string (name, value) pairs that the policy can parse to arguments
        :param board_size: the size of the game board (height, width)
        :param logq: a queue for message logging through the game
        :param game_duration: the expected duration of the game in turns
        :param score_scope: the amount of rounds at the end of the game which count towards the score
        """

        self.id = id
        self.len = 0
        self.policy_class = policy
        self.round = 0
        self.unresponsive_count = 0
        self.too_slow = False

        self.sq = mp.Queue()
        self.aq = mp.Queue()
        self.mq = mp.Queue()
        self.logq = logq
        self.policy = policy(policy_args, board_size, self.sq, self.aq, self.mq, logq, id, game_duration, score_scope)
        self.policy.daemon = True
        self.policy.start()


    def handle_state(self, round, prev_state, prev_action, reward, new_state):
        """
        given the new state and previous state-action-reward, pass the information
        to the policy for action selection and/or learning.
        """

        self.round = round
        clear_q(self.sq)  # remove previous states from queue if they weren't handled yet
        self.sq.put((round, prev_state, prev_action, reward, new_state, self.too_slow))


    def get_action(self):
        """
        get waiting action from the policy's action queue. if there is no action
        in the queue, pick 'F' and log the unresponsiveness error.
        :return: action from {'R','L','F'}.
        """
        try:
            round, action = self.aq.get_nowait()
            if round != self.round:
                raise queue.Empty()
            elif action not in Policy.Policy.ACTIONS:
                self.logq.put((str(self.id), "ERROR", ILLEGAL_MOVE + str(action)))
                raise queue.Empty()
            else:
                self.too_slow = False
                self.unresponsive_count = 0

        except queue.Empty:
            self.unresponsive_count += 1
            action = Policy.Policy.DEFAULT_ACTION
            if self.unresponsive_count <= UNRESPONSIVE_POLICY_THRESHOLD:
                self.logq.put((str(self.id), "ERROR", NO_RESPONSE + str(self.unresponsive_count) + " in a row!"))
            else:
                self.logq.put((str(self.id), "ERROR", UNRESPONSIVE_PLAYER))
                self.unresponsive_count = TOO_SLOW_POLICY_THRESHOLD
            if self.unresponsive_count > TOO_SLOW_POLICY_THRESHOLD:
                self.too_slow = True

        clear_q(self.aq)  # clear the queue from unhandled actions
        return action


    def shutdown(self):
        """
        shutdown the agent in the end of the game. the function asks the agent
        to save it's model and returns the saved model, which needs to be a data
        structure that can be pickled.
        :return: the model data structure.
        """

        clear_q(self.sq)
        clear_q(self.aq)
        self.sq.put(None)  # shutdown signal
        self.policy.join()
        return


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
        self.item_count = 0
        # self.board = EMPTY_CELL_VAL * np.ones(self.board_size, dtype=int)
        self.previous_board = None

        # initialize players:
        self.rewards, self.players, self.scores, self.directions, self.actions, self.growing, self.size, self.previous_heads = \
            [], [], [], [], [], [], [], []
        for i, (policy, pargs) in enumerate(self.policies):
            self.rewards.append(0)
            self.actions.append(None)
            self.previous_heads.append(None)
            self.scores.append([0])
            self.players.append(Agent(i, policy, pargs, self.board_size, self.logq, self.game_duration, self.score_scope))
            color = pargs['color'] if 'color' in pargs else SN_COLOR
            player_size, growing, direction = self.init_player(i, color)
            self.size.append(player_size)
            self.growing.append(growing)
            self.directions.append((direction))








        # wait for player initialization (Keras loading time):
        time.sleep(self.player_init_time)


    def init_player(self, id, color, reset=False):

        # initialize the position and direction of the player:
        dir = np.random.choice(list(Policy.Policy.TURNS.keys()))
        shape = (1, 3) if dir in ['W', 'E'] else (3, 1)
        first = self.game_gui.get_empty_slot(shape)
        sec = self.game_gui.move_step(dir, first[0], first[1])
        assert self.game_gui.get_board_array()[sec[0], sec[1]] == EMPTY_CELL_VAL
        player_size = 2
        growing = True

        snake_gui = Snake(self.game_gui, id,  color)
        positions = [first, sec]

        snake_gui.set_positions(positions)
        snake_gui.set_direction(dir)
        self.game_gui.add_sneak(snake_gui)
        if reset:
            snake_gui.start()



        return player_size, growing, dir


    # def reset_player(self, id):
    #
    #     positions = np.array(np.where(self.game_gui.get_board_array() == id))
    #     self.game_gui.delete_snake(id)
    #
    #     # turn parts of the corpse into food:
    #     food_n = np.random.binomial(positions.shape[1], self.food_ratio)
    #     if self.item_count + food_n < self.max_item_density * np.prod(self.board_size):
    #         subidx = np.array(np.random.choice(positions.shape[1], size=food_n, replace=False))
    #         if len(subidx) > 0:
    #             randfood = np.random.choice(list(FOOD_GROWING_MAP.keys()), food_n)
    #             for i,idx in enumerate(subidx):
    #                 # self.board[positions[0,idx],positions[1,idx]] = randfood[i]
    #                 self.game_gui.add_fruit(positions[0,idx], positions[1,idx], randfood[i])
    #             self.item_count += food_n
    #
    #     return self.init_player(id, self.game_gui.get_sneak_color(id), True)


    # def randomize(self):
    #     if np.random.rand(1) < self.random_food_prob:
    #         if self.item_count < self.max_item_density * np.prod(self.board_size):
    #             randfood = np.random.choice(list(FOOD_GROWING_MAP.keys()), 1)
    #             pos = self.game_gui.get_empty_slot((1, 1))
    #             self.game_gui.add_fruit(pos[0], pos[1], randfood[0])
    #             self.item_count += 1


    def move_snake(self, id, action):

        # delete the tail if the snake isn't growing:
        growing = True
        if self.growing[id] > 0:
            self.growing[id] -= 1
            self.size[id] += 1
        else:
            growing = False

        print(growing)


        # move the head:
        if action != 'F':  # turn in the relevant direction
            self.directions[id] = Policy.Policy.TURNS[self.directions[id]][action]


        self.game_gui.move_sneak(id, self.directions[id], growing)


    def play_a_round(self):

        # randomize the players:
        pperm = np.random.permutation([(i,p) for i, p in enumerate(self.players)])

        # distribute states and rewards on previous round
        board = self.game_gui.get_board_array()
        for i, p in pperm:
            x, y = self.game_gui.get_snake(p.id).get_head()
            current_head = (Position((x,y), self.board_size), self.directions[p.id])
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
            action = actions[p]
            self.actions[p.id] = action
            head = self.game_gui.get_snake(p.id).get_head()
            move_to = self.game_gui.move_step(Policy.Policy.TURNS[self.directions[p.id]][action], head[0], head[1])
            # reset the player if he died:
            if board[move_to[0], move_to[1]] != EMPTY_CELL_VAL and board[move_to[0], move_to[1]] not in FOOD_GROWING_MAP:
                self.game_gui.reset_player(p.id)
                player_size, growing, dir = self.init_player(p.id, self.game_gui.get_sneak_color(p.id), True)
                self.size[p.id] = player_size
                self.growing[p.id] = growing
                self.directions[p.id] = dir
                self.rewards[p.id] = DEATH_PENALTY
                self.scores[p.id].append(self.rewards[p.id])


            # otherwise, move the player on the board:
            else:
                self.rewards[p.id] = 0
                if board[move_to[0], move_to[1]] in FOOD_GROWING_MAP.keys():
                    self.rewards[p.id] += FOOD_SCORE_MAP[board[move_to[0], move_to[1]]]
                    self.growing[p.id] += FOOD_GROWING_MAP[board[move_to[0], move_to[1]]]  # start growing
                    self.item_count -= 1

                self.move_snake(p.id, action)
                self.scores[p.id].append(self.rewards[p.id])

        # update the food on the board:
        self.item_count = self.game_gui.randomize()
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
