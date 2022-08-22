import numpy as np


class Game:
    def __init__(self, map_h=1000, map_w=1000):
        self.map_h = map_h
        self.map_w = map_w

        self.elevation = np.ones(map_h, map_w)
        self.elevation[map_h//3*2:] = -1
    
    def render(self, pos_x, pos_y, scale):
        return self.elevation