
from gui_board.BoardConstants import *
from gui_board.Shape import Block
from Constants import *
from tkinter import Canvas
class Snake:
    """a snake keeps track of its body parts"""
    def __init__(self, can, id, color, growing_init):
        """initial position chosen by me"""
        self.id = id
        self.can = can
        self.direction = None
        self.current_movement = None
        self.score_manager = None
        self.color = color
        self.positions = None
        self.growing = growing_init


    def update_score(self, score_manager):
        self.score_manager = score_manager

    def change_direction(self, direction, growing):
        if direction is not None: self.direction = direction
        if self.direction is not None:
            self.current_movement = Movement(self, self.can, self.direction)
            self.current_movement.begin(growing)

    def set_positions(self, positions):
        self.positions = positions
    def set_direction(self, direction):
        self.direction = direction


    def start(self):
        # a,b = self.can.get_random_empty_idx()# random places with gui locations
        self.blocks = []
        for pos in self.positions:
            a, b = self.can.get_position_in_board(pos[0], pos[1])
            self.blocks.append(Block(self.can, a, b, self.id, self.color))
        self.blocks[-1].set_poly_fill('white')



    def delete_from_board(self):
        if self.current_movement is not None:
            self.current_movement.stop()
        else:
            print("in")
        for block in self.blocks:
            block.delete()

    def delete(self):
        self.delete_from_board()
        if self.score_manager is not None:
            self.score_manager.reset_score(self.id)

    def move_sneak(self, action):
        growing = True
        if self.growing > 0:
            self.growing -= 1
        else:
            growing = False
        new_direction = self.direction
        if action != 'F':
            new_direction = TURNS[new_direction][action]
        self.change_direction(new_direction, growing)

    def _move(self, path, growing):
        """an elementary step consisting of putting the tail of the snake in the first position"""
        a = (self.blocks[-1].x + STEP * path[0]) % WD
        b = (self.blocks[-1].y + STEP * path[1]) % HT

        self.blocks[-1].set_poly_fill(self.color)

        value_in_cell = self.can.get_board_cell_value(a, b)
        if value_in_cell in self.can.fruits.keys():  # check if we find food
            self.can.fruits[int(value_in_cell)].delete()
            self.blocks.append(Block(self.can, a, b, self.id, self.color))
        elif growing:
            self.blocks.append(Block(self.can, a, b, self.id, self.color))
        else:
            self.blocks[0].modify(a, b)
            self.blocks = self.blocks[1:] + [self.blocks[0]]

        self.blocks[-1].set_poly_fill('white')






    def get_head(self):
        return self.can.get_position_in_array(self.blocks[-1].x, self.blocks[-1].y)











class KeyBoardSnake(Snake):

    def __init__(self, can, direction, id, color):
        super().__init__(can, direction, id, color)
        self.can.boss.bind("<Key>", self.redirect)


    def redirect(self, event):
        """taking keyboard inputs and moving the snake accordingly"""
        if 1 == self.can.running and \
                event.keysym in AXES.keys() and \
                AXES[event.keysym] != AXES[self.direction]:
            self.current_movement.flag = 0
            self.change_direction(event.keysym)


class Movement:
    """object that enters the snake into a perpetual state of motion in a predefined direction"""
    def __init__(self, sneak, can, direction):
        self.flag = 1
        self.can = can
        self.direction = direction
        self.sneak = sneak

    def begin(self, growing):
        """start the perpetual motion"""
        if self.flag > 0:
            self.sneak._move(DIRECTIONS[self.sneak.direction], growing)


    def stop(self):
        """stop the perpetual movement"""
        self.flag = 0



