import time
import datetime
import gzip
import multiprocessing as mp

from GameComponants import *
from gui.gui_board import init_board_gui, Board, start_board
from config import parse_args


def days_hours_minutes_seconds(td):
    """
    parse time for logging.
    """
    return td.days, td.seconds//3600, (td.seconds//60)%60, td.seconds%60


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

        # check that the number of players is OK:
        self.n = len(self.policies)
        assert self.n <= MAX_PLAYERS, "Too Many Players!"
        self.round = 0
        self.previous_board = None

        # init logger
        self.logq = mp.Queue()
        self.logger = mp.Process(target=self.log, args=(self.logq, self.log_file))
        self.logger.start()


        # initialize players:
        self.snakes = {}
        self.actions = []
        self.players = []
        self.previous_heads = []
        for i, (policy, pargs) in enumerate(self.policies):
            color = pargs['color'] if 'color' in pargs else SN_COLOR
            self.actions.append(None)
            self.previous_heads.append(None)
            self.players.append(Agent(i, policy, pargs, self.board_size, self.logq, self.game_duration, self.score_scope, self.snakes))

            self.snakes[i] = Snake(i, color, self.init_player_size)

        # gui board:
        self.tk_root, self.board = init_board_gui(self.board_size[0], self.board_size[1], FOOD_MAP,
                                                        self.random_food_prob, self.max_item_density, self.obstacle_density, self.run, self.snakes)


        # wait for player initialization (Keras loading time):
        time.sleep(self.player_init_time)




    def play_a_round(self):

        # randomize the players:
        pperm = np.random.permutation([(i,p) for i, p in enumerate(self.players)])

        # distribute states and rewards on previous round
        for i, p in pperm:
            snake = self.board.snakes[i]["snake_obj"]
            current_head = (snake.positions[-1], snake.direction)
            if self.previous_board is None:
                p.handle_state(self.round, None, self.actions[p.id], snake.current_score, (self.board.board_array, current_head))
            else:
                p.handle_state(self.round, (self.previous_board, snake.previous_head), self.actions[p.id], snake.current_score, (self.board.board_array, current_head))
            snake.previous_head = current_head
        self.previous_board = np.copy(self.board.board_array)

        # wait and collect actions
        time.sleep(self.policy_action_time)
        actions = {p: p.get_action() for _, p in pperm}

        if self.round % LEARNING_TIME == 0 and self.round > 5:
            time.sleep(self.policy_learn_time)

        # get the interactions of the players with the board:
        for _, p in pperm:
            action = actions[p]
            self.actions[p.id] = action
            snake = self.snakes[p.id]
            new_direction = TURNS[snake.direction][action]
            new_cell_pos, new_cell_type, new_cell_value, will_grow = Board.move_snake(snake, new_direction, self.board.board_array,
                                                             self.board.snakes.keys(), self.board.fruits_types)
            if new_cell_type == "collision":
                self.board.handle_collision(snake)
            else:
                if new_cell_type == "food":
                    self.board.delete_food_from_board(new_cell_pos)
                self.board.move_gui_board_snake(snake, will_grow)

            snake.all_scores.append(snake.current_score)
            snake.current_score = 0



        # update the food on the board:
        self.board.add_food()
        self.round += 1



    def run(self):
        try:
            r = 0
            while r < self.game_duration:
                r += 1
                time.sleep(0.2)
                if r % STATUS_UPDATE == 0:
                    print("At Round " + str(r) + " the scores are:")

                    self.board.update_scores(r, self.score_scope)

                self.play_a_round()

        finally:
            print("end")
            # output = [','.join(['game_id','player_id','policy','score'])]
            # game_id = str(abs(id(self)))
            # for p, s in zip(self.players, self.scores):
            #     p.shutdown()
            #     pstr = str(p.policy).split('<')[1].split('(')[0]
            #     scope = np.min([len(s), self.score_scope])
            #     p_score = np.mean(s[-scope:])
            #     oi = [game_id, str(p.id), pstr, str("{0:.4f}".format(p_score))]
            #     output.append(','.join(oi))
            #
            # with open(self.output_file, 'w') as outfile:
            #     outfile.write('\n'.join(output))
            # self.logq.put(None)
            # self.logger.join()





if __name__ == '__main__':
    g = Game(parse_args())
    start_board(g.tk_root, g.board)

