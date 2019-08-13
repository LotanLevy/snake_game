
from tkinter import Canvas
from gui_board.BoardConstants import *



class Shape:
    """This is a template to make obstacles and snake body parts"""
    def __init__(self, can, a, b, kind, id, color):
        self.can = can
        self.id = id
        self.x, self.y = a, b
        self.kind = kind
        self.color = color
        if kind == SN:
            self.ref = Canvas.create_rectangle(self.can,
                                               a - SN_SIZE, b - SN_SIZE,
                                               a + SN_SIZE, b + SN_SIZE,
                                               fill=self.color,
                                               width=2)
            can.update_array(a, b, self.id)
        elif kind == OB:
            self.ref = Canvas.create_oval(self.can,
                                          a - OB_SIZE, b - OB_SIZE,
                                          a + SN_SIZE, b + SN_SIZE,
                                          fill=self.color,
                                          width=2)
            can.update_array(a, b, self.id)



    def modify(self, a, b):
        self.can.update_array(self.x, self.y, EMPTY_BOARD_FIELD)
        self.x, self.y = a, b
        self.can.coords(self.ref,
                        a - SIZE[self.kind], b - SIZE[self.kind],
                        a + SIZE[self.kind], b + SIZE[self.kind])
        self.can.update_array(a, b, self.id)


    def delete(self):
        self.can.delete(self.ref)
        self.can.update_array(self.x, self.y, EMPTY_BOARD_FIELD)


class Obstacle(Shape):
    """snake food"""
    def __init__(self, can, id, color, type_id, x, y):
        """only create the obstacles where there is no snake body part"""
        self.can = can
        self.type_id = type_id
        self.x, self.y = self.can.get_position_in_board(x, y)
        super().__init__(can, self.x, self.y, OB, id, color)
        self.can.item_count += 1

    def get_position(self):
        return self.x, self.y

    def delete(self, from_board_dict=True):
        super().delete()
        if from_board_dict:
            del self.can.fruits[self.id]
            self.can.item_count -= 1






class Block(Shape):
    """snake body part"""
    def __init__(self, can, a, y, id, color):
        super().__init__(can, a, y, SN, id, color)

#
# class Snake:
#     """a snake keeps track of its body parts"""
#     def __init__(self, can, id, color):
#         """initial position chosen by me"""
#         self.id = id
#         self.can = can
#         self.direction = None
#         self.current_movement = None
#         self.score_manager = None
#         self.color = color
#         self.positions = None
#
#     def update_score(self, score_manager):
#         self.score_manager = score_manager
#
#     def change_direction(self, direction, growing):
#         if direction is not None: self.direction = MAIN_DIRECTION_MAP[direction]
#         if self.direction is not None:
#             self.current_movement = Movement(self, self.can, self.direction)
#             self.current_movement.begin(growing)
#
#     def set_positions(self, positions):
#         self.positions = positions
#     def set_direction(self, direction):
#         self.direction = MAIN_DIRECTION_MAP[direction]
#
#
#     def start(self):
#         # a,b = self.can.get_random_empty_idx()# random places with gui locations
#         self.blocks = []
#         for pos in self.positions:
#             a, b = self.can.get_position_in_board(pos[0], pos[1])
#             self.blocks.append(Block(self.can, a, b, self.id, self.color))
#
#
#
#     def delete_from_board(self):
#         if self.current_movement is not None:
#             self.current_movement.stop()
#         else:
#             print("in")
#         for block in self.blocks:
#             block.delete()
#
#     def delete(self):
#         self.delete_from_board()
#         if self.score_manager is not None:
#             self.score_manager.reset_score(self.id)
#
#     def move(self, path, growing):
#         """an elementary step consisting of putting the tail of the snake in the first position"""
#         a = (self.blocks[-1].x + STEP * path[0]) % WD
#         b = (self.blocks[-1].y + STEP * path[1]) % HT
#
#         value_in_cell = self.can.get_board_cell_value(a, b)
#         if value_in_cell in self.can.fruits.keys():  # check if we find food
#             self.can.fruits[int(value_in_cell)].delete()
#             self.blocks.append(Block(self.can, a, b, self.id, self.color))
#         elif growing:
#             self.blocks.append(Block(self.can, a, b, self.id, self.color))
#         else:
#             self.blocks[0].modify(a, b)
#             self.blocks = self.blocks[1:] + [self.blocks[0]]
#
#
#

#
# class KeyBoardSnake(Snake):
#
#     def __init__(self, can, direction, id, color):
#         super().__init__(can, direction, id, color)
#         self.can.boss.bind("<Key>", self.redirect)
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
#     def begin(self, growing):
#         """start the perpetual motion"""
#         if self.flag > 0:
#             self.sneak.move(DIRECTIONS[self.sneak.direction], growing)
#
#
#     def stop(self):
#         """stop the perpetual movement"""
#         self.flag = 0
#
#



