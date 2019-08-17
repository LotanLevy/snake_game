
from tkinter import Canvas, StringVar, Label
from GameConfigurations import *
from abc import ABC, abstractmethod

class GuiComponent(ABC):
    def __init__(self, tk_root):
        self.tk_root = tk_root

    @abstractmethod
    def delete(self):
        pass


class BoardScore(GuiComponent):
    def __init__(self, tk_root, scoreboard, score_title_format):
        super().__init__(tk_root)
        self._score_title_format = score_title_format

        self.round_score = StringVar(self.tk_root, '0')
        self.round = StringVar(self.tk_root, (self._score_title_format + " in round {}").format(0))

        self.title = Label(scoreboard, textvariable=self.round)
        self.score = Label(scoreboard, textvariable=self.round_score)
        self.title.grid()
        self.score.grid()


    def update(self, new_score, round):
        self.round_score.set(str(new_score))
        self.round.set((self._score_title_format + " in round {}").format(round))
        print((self._score_title_format + " in round {}").format(round))
        print(new_score)


    def get_score(self):
        return float(self.round_score.get())

    def delete(self):
        self.title.destroy()
        self.score.destroy()






class BoardShape:
    """This is a template to make obstacles and snake body parts"""
    def __init__(self, canvas, a, b, kind, id, color):
        self.canvas = canvas
        self.id = id
        self.x, self.y = a, b
        self.kind = kind
        self.color = color
        if kind == SN:
            self.ref = Canvas.create_rectangle(self.canvas,
                                               self.x - BLOCK_SIZE, self.y - BLOCK_SIZE,
                                               self.x + BLOCK_SIZE, self.y + BLOCK_SIZE,
                                               fill=self.color,
                                               width=2)
        elif kind == OB:
            self.ref = Canvas.create_oval(self.canvas,
                                          self.x - BLOCK_SIZE, self.y - BLOCK_SIZE,
                                          self.x + BLOCK_SIZE, self.y + BLOCK_SIZE,
                                          fill=self.color,
                                          width=2)



    def modify(self, a, b):
        # self.can.update_array(self.x, self.y, EMPTY_BOARD_FIELD)
        self.x, self.y = a, b
        self.canvas.coords(self.ref,
                        a - BLOCK_SIZE, b - BLOCK_SIZE,
                        a + BLOCK_SIZE, b + BLOCK_SIZE)
        # self.can.update_array(a, b, self.id)

    def set_poly_fill(self, color):
        # if poly exists then you can change fill
        if self.ref:
            self.canvas.itemconfig(self.ref, fill=color)


    def delete(self):
        self.canvas.delete(self.ref)
        # self.tk_root.update_array(self.x, self.y, EMPTY_BOARD_FIELD)


class Fruit(BoardShape):
    """snake food"""
    def __init__(self, tk_root , id, x, y, color, type_id):
        """only create the obstacles where there is no snake body part"""
        self.type_id = type_id
        self.x, self.y = PIXEL * (2 * y + 1), PIXEL * (2 * x + 1)
        super().__init__(tk_root, self.x, self.y, OB, id, color)

    def get_position(self):
        return self.x, self.y

    def delete(self):
        super().delete()


class Obstacle:
    def __init__(self, canvas, id, block_positions, color):
        self.positions = block_positions
        self.blocks = []
        self.canvas = canvas
        self.id = id
        self.color = color

    def start(self):
        for pos in self.positions:
            x, y = PIXEL * (2*pos[1] + 1), PIXEL * (2*pos[0] + 1)
            self.blocks.append(Block(self.canvas, x, y, self.id, self.color))




class Block(BoardShape):
    """snake body part"""
    def __init__(self, tk_root, a, y, id, color):
        super().__init__(tk_root, a, y, SN, id, color)


class BoardSnake:
    """a snake keeps track of its body parts"""
    def __init__(self, tk_root,  id, color, head_color, positions, board_shape):
        """initial position chosen by me"""
        self.id = id
        self.tk_root = tk_root
        self.color = color
        self.head_color = head_color
        self.blocks = []
        self.init_positions = positions
        self.board_shape = board_shape

    def start(self):
        self.delete()
        self._set_positions()

    def _set_positions(self):
        for pos in self.init_positions:
            x, y = PIXEL * (2*pos[1] + 1), PIXEL * (2*pos[0] + 1)
            self.blocks.append(Block(self.tk_root, x, y, self.id, self.color))
        # self.blocks[-1].set_poly_fill(self.head_color)

    def move(self, dir, growing):
        dir_vals = DIRECTIONS[dir]
        x = (self.blocks[-1].x + STEP * dir_vals[1]) % self.board_shape[0]
        y = (self.blocks[-1].y + STEP * dir_vals[0]) % self.board_shape[1]

        self.blocks.append(Block(self.tk_root, x, y, self.id, self.color))
        if not growing:
            to_delete = self.blocks[0]
            self.blocks = self.blocks[1:]
            to_delete.delete()

    def delete(self):
        for block in self.blocks:
            block.delete()


    #
    # def change_direction(self, direction):
    #     self.direction = direction
    #     self.current_movement = Movement(self, self.can, self.direction)
    #     self.begin_movement()
    #
    # def start(self):
    #     a,b = self.can.get_random_empty_idx()# random places with gui locations
    #     self.blocks = [Block(self.can, a, b, self.id, self.color)]
    #     self.current_movement = Movement(self, self.can, self.direction)
    #
    # def begin_movement(self):
    #     self.current_movement.begin()
    #
    #
    # def delete(self):
    #     for block in self.blocks:
    #         block.delete()
    #
    # def manage_collision(self):
    #     self.start()
    #     self.begin_movement()
    #     if self.score_manager is not None:
    #         self.score_manager.update_score(self.id, COLLISION_PUNISHMENT)
    #
    # def move(self, path):
    #     """an elementary step consisting of putting the tail of the snake in the first position"""
    #     a = (self.blocks[-1].x + STEP * path[0]) % WD
    #     b = (self.blocks[-1].y + STEP * path[1]) % HT
    #
    #     value_in_cell = self.can.get_board_cell_value(a, b)
    #     if value_in_cell in self.can.fruits.keys():  # check if we find food
    #         if self.score_manager is not None:
    #             fruit_obj = self.can.fruits[value_in_cell]
    #             self.score_manager.update_score(self.id, self.can.fruits_types[fruit_obj.type_id]["score"])
    #         self.can.fruits[int(value_in_cell)].delete()
    #         self.blocks.append(Block(self.can, a, b, self.id, self.color))
    #         self.can.add_fruit()
    #     elif value_in_cell in self.can.snakes.keys():
    #         self.can.snakes[value_in_cell].delete_from_board()
    #         self.delete_from_board()
    #         self.can.snakes[value_in_cell].manage_collision()
    #         self.manage_collision()
    #     elif [a, b] in [[block.x, block.y] for block in self.blocks]:  # check if we hit a body part
    #         self.can.clean()
    #     else:
    #         self.blocks[0].modify(a, b)
    #         self.blocks = self.blocks[1:] + [self.blocks[0]]


#
#
# class KeyBoardSnake(BoardSnake):
#
#     def __init__(self, can, direction, id, color):
#         super().__init__(can, direction, id, color)
#         self.can.tk_root.bind("<Key>", self.redirect)
#
#
#     def redirect(self, event):
#         """taking keyboard inputs and moving the snake accordingly"""
#         if 1 == self.can.running and \
#                 event.keysym in AXES.keys() and \
#                 AXES[event.keysym] != AXES[self.direction]:
#             self.current_movement.flag = 0
#             self.change_direction(event.keysym)
#
#
# class Movement:
#     """object that enters the snake into a perpetual state of motion in a predefined direction"""
#     def __init__(self, sneak, can, direction):
#         self.flag = 1
#         self.can = can
#         self.direction = direction
#         self.sneak = sneak
#
#     def begin(self):
#         """start the perpetual motion"""
#         if self.flag > 0:
#             self.sneak.move(DIRECTIONS[self.sneak.direction])
#             self.can.after(REFRESH_TIME, self.begin)
#
#     def stop(self):
#         """stop the perpetual movement"""
#         self.flag = 0





