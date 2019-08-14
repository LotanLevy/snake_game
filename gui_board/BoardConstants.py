

# constants that go in the making of the grid used for the snake's movment
GRADUATION = 40
PIXEL = 10
STEP = 2 * PIXEL
WD = PIXEL * GRADUATION
HT = PIXEL * GRADUATION
# constants that go into specifying the shapes' sizes
OB_SIZE_FACTOR = 1
SN_SIZE_FACTOR = 1
OB_SIZE = PIXEL * OB_SIZE_FACTOR
SN_SIZE = PIXEL * SN_SIZE_FACTOR
# color constants
BG_COLOR = 'black'
OB_COLOR = 'red'
SN_COLOR = 'white'
# a dictionary to ease access to a shape's type in the Shape class
SN = 'snake'
OB = 'obstacle'
SIZE = {SN: SN_SIZE, OB: OB_SIZE}
# constants for keyboard input
UP = 'Up'
DOWN = 'Down'
RIGHT = 'Right'
LEFT = 'Left'
# a dictionary to ease access to 'directions'
# DIRECTIONS = {UP: [0, -1], DOWN: [0, 1], RIGHT: [1, 0], LEFT: [-1, 0]}
DIRECTIONS = {"N": [0, -1], 'S': [0, 1], 'E': [1, 0], "W": [-1, 0]}

MAIN_DIRECTION_MAP = {'E':RIGHT, "W":LEFT, "N":UP, 'S':DOWN}
AXES = {UP: 'Vertical', DOWN: 'Vertical', RIGHT: 'Horizontal', LEFT: 'Horizontal'}
# refresh time for the perpetual motion
REFRESH_TIME = 400
EMPTY_BOARD_FIELD= -1
COLLISION_PUNISHMENT = -5