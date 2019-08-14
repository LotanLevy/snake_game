
PLAYER_INIT_TIME = 60


ILLEGAL_MOVE = "Illegal Action: the default action was selected instead. Player tried action: "
NO_RESPONSE = "No Response: player took too long to respond with action. This is No Response #"
UNRESPONSIVE_PLAYER = "Unresponsive Player: the player hasn't responded in too long... SOMETHING IS WRONG!!"

STATUS_UPDATE = 100
TOO_SLOW_POLICY_THRESHOLD = 3
UNRESPONSIVE_POLICY_THRESHOLD = 50
LEARNING_FREQ = 5

EMPTY_CELL_VAL = -1
MAX_PLAYERS = 5
FOOD_GROWING_MAP = {6:1, 7:1, 8:1}
FOOD_SCORE_MAP = {6:2, 7:5, 8:-1}
FOOD_COLOR_MAP = {6:(FOOD_SCORE_MAP[6], 'purple'), 7:(FOOD_SCORE_MAP[7], 'yellow'), 8:(FOOD_SCORE_MAP[8], 'green')}
DEATH_PENALTY = -5

DEFAULT_ACTION = 'F'
ACTIONS = ['L',  # counter clockwise (left)
           'R',  # clockwise (right)
           'F']  # forward
TURNS = {
    'N': {'L': 'W', 'R': 'E', 'F': 'N'},
    'S': {'L': 'E', 'R': 'W', 'F': 'S'},
    'W': {'L': 'S', 'R': 'N', 'F': 'W'},
    'E': {'L': 'N', 'R': 'S', 'F': 'E'}
}
