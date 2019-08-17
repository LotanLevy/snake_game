
from abc import abstractmethod
import numpy as np
from GameConfigurations import *
import queue
import policies.Policy as base_policy
import multiprocessing as mp

class HasScore:
    @abstractmethod
    def get_score(self):
        pass

class Snake(HasScore):
    def __init__(self, id, color, init_player_size):
        self.id = id
        self.color = color
        self.direction = None
        self.growing = 0
        self.current_score = 0
        self.all_scores = []
        self.size = 0
        self.positions = []
        self.init_player_size = init_player_size
        self.previous_head = None

    def get_gui_representation(self):
        return "snake {} - ({})".format(self.id, self.color)

    def __str__(self):
        return "snake {}".format(self.id)

    def update_score(self, reward):
        self.current_score = reward


    def get_score(self):
        return self.current_score

    def get_mean_score_in_scope(self, scope_length):
        scope = np.min([len(self.all_scores), scope_length])
        return np.mean(self.all_scores[-scope:])

    def move(self, pos, board_array):
        self.positions.append(pos)
        board_array[pos[0], pos[1]] = self.id
        if not self.growing:
            tail = self.positions[0]
            self.positions = self.positions[1:]
            board_array[tail[0], tail[1]] = EMPTY_BOARD_FIELD
            return False
        else:
            self.growing -= 1
            return True





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
        return self + DIRECTIONS[dir]





class Agent(object):
    SHUTDOWN_TIMEOUT = 60 # seconds until policy is considered unresponsive

    def __init__(self, id, policy, policy_args, board_size, logq, game_duration, score_scope, snakes):
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
        # self.logq = logq
        self.policy = policy(policy_args, board_size, self.sq, self.aq, self.mq, logq, id, game_duration, score_scope, snakes)
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
            elif action not in base_policy.Policy.ACTIONS:
                # self.logq.put((str(self.id), "ERROR", ILLEGAL_MOVE + str(action)))
                raise queue.Empty()
            else:
                self.too_slow = False
                self.unresponsive_count = 0

        except queue.Empty:
            self.unresponsive_count += 1
            action = base_policy.Policy.DEFAULT_ACTION
            if self.unresponsive_count <= UNRESPONSIVE_THRESHOLD:
                pass
                # self.logq.put((str(self.id), "ERROR", NO_RESPONSE + str(self.unresponsive_count) + " in a row!"))
            else:
                # self.logq.put((str(self.id), "ERROR", UNRESPONSIVE_PLAYER))
                self.unresponsive_count = TOO_SLOW_THRESHOLD
            if self.unresponsive_count > TOO_SLOW_THRESHOLD:
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


def clear_q(q):
    """
    given a queue, empty it.
    """
    while not q.empty():
        try: q.get_nowait()
        except queue.Empty: break





