

import queue
import multiprocessing as mp
from Constants import *




class Agent(object):
    SHUTDOWN_TIMEOUT = 60 # seconds until policy is considered unresponsive

    def __init__(self, id, policy, policy_args, board_size, logq, game_duration, score_scope, adversaries_ids, fruits_ids):
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
        self.policy = policy(policy_args, board_size, self.sq, self.aq, self.mq, logq, id, game_duration, score_scope, adversaries_ids, fruits_ids)
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
            elif action not in ACTIONS:
                self.logq.put((str(self.id), "ERROR", ILLEGAL_MOVE + str(action)))
                raise queue.Empty()
            else:
                self.too_slow = False
                self.unresponsive_count = 0

        except queue.Empty:
            self.unresponsive_count += 1
            action = DEFAULT_ACTION
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


def clear_q(q):
    """
    given a queue, empty it.
    """
    while not q.empty():
        try: q.get_nowait()
        except queue.Empty: break
