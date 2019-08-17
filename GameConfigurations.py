
# constants that go in the making of the grid used for the snake's movment
GRADUATION = 40
PIXEL = 10
STEP = 2 * PIXEL
WD = PIXEL * 60
HT = PIXEL * GRADUATION
# constants that go into specifying the shapes' sizes
# OB_SIZE_FACTOR = 1
# SN_SIZE_FACTOR = 1
# OB_SIZE = PIXEL * OB_SIZE_FACTOR
BLOCK_SIZE = PIXEL
# color constants
BG_COLOR = 'black'
OB_COLOR = 'gray'
SN_COLOR = 'white'
# a dictionary to ease access to a shape's type in the Shape class
SN = 'snake'
OB = 'obstacle'
# SIZE = {SN: BLOCK_SIZE, OB: OB_SIZE}
# constants for keyboard input
UP = 'Up'
DOWN = 'Down'
RIGHT = 'Right'
LEFT = 'Left'
# a dictionary to ease access to 'directions'
DIRECTIONS = {'W': [0, -1], 'E': [0, 1], 'S': [1, 0], 'N': [-1, 0]}
TURNS = {
    'N': {'L': 'W', 'R': 'E', 'F': 'N'},
    'S': {'L': 'E', 'R': 'W', 'F': 'S'},
    'W': {'L': 'S', 'R': 'N', 'F': 'W'},
    'E': {'L': 'N', 'R': 'S', 'F': 'E'}
}
# DIRECTIONS = {UP: [0, -1], DOWN: [0, 1], RIGHT: [1, 0], LEFT: [-1, 0]}
AXES = {UP: 'Vertical', DOWN: 'Vertical', RIGHT: 'Horizontal', LEFT: 'Horizontal'}
# refresh time for the perpetual motion
REFRESH_TIME = 100
EMPTY_BOARD_FIELD= -1
COLLISION_PUNISHMENT = -5
STATUS_UPDATE = 100
FOOD_MAP = {6:{"growing":1, "score":2, "color": "red"},
            7:{"growing":3, "score":5, "color": "orange"},
            8:{"growing":0, "score":-1, "color": "blue"}}


EMPTY_VAL = -1
MAX_PLAYERS = 5
OBSTACLE_VAL = 5
REGULAR_RENDER_MAP = {EMPTY_VAL: ' ', OBSTACLE_VAL: '+'}

THE_DEATH_PENALTY = -5

ILLEGAL_MOVE = "Illegal Action: the default action was selected instead. Player tried action: "
NO_RESPONSE = "No Response: player took too long to respond with action. This is No Response #"
PLAYER_INIT_TIME = 60
UNRESPONSIVE_PLAYER = "Unresponsive Player: the player hasn't responded in too long... SOMETHING IS WRONG!!"

TOO_SLOW_THRESHOLD = 3
UNRESPONSIVE_THRESHOLD = 50
LEARNING_TIME = 5