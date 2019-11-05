import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Grid(object):
    def __init__(self,x:int =None,y:int=None,type:int=0,
                reward:float=0.0):
        self.x = x
        self.y = y
        self.type = type  ##define the Grid value
class GridMatrix(object):

    '''
    w,h,type,reward
    '''
    def __init__(self,n_width:int,n_height:int,default_type:int =0,default_reward:float=0.0):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width*n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self.reset()
        ## grids is  a list of grid obj
    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,y,self.default_type,self.default_reward)) 
    def get_grid(self,x,y=None):
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert(xx>=0 and yy>=0 and xx < self.n_width and yy < self.n_height),\
                "任意坐标值应在合理区间"
        index = yy * self.n_width + xx
        return self.grids[index]
    def set_reward(self, x, y, reward):
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.reward = reward
        else:
            raise("grid doesn't exist")
    def set_type(self, x, y, type):
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.type = type
        else:
            raise("grid doesn't exist")
    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.reward

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.type

class GridEnv(gym.Env):
    metadata = {
        'render.modes':['human','rgb_array'],
        'video.frams_per_second':30
    }
    def __init__(self,n_width:int = 10,n_height:int =7,
                u_size = 40,default_reward:float = 0,
                default_type = 0):
        self.u_size = u_size 
        self.n_width = n_width
        self.width = u_size *n_width
        self.n_height = u_size*n_height
        self.height = u_size*n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()

        self.grids = GridMatrix(n_width = self.n_width,
            n_height = self.n_height,
            default_reward=  self.default_reward,
            default_type = self.default_type,)
        self.reward = 0
        self.action = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)
        self.ends = [(7,3)] 
        self.start = [(0,3)] 
        self.rewards = []
        self.refresh_setting()
        self.viewer = None
        self.seed()
        self.reset()
    
    def _adjust_size(self):
        '''调整场景尺寸适合最大宽度、高度不超过800
        '''
        pass
    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)  
        return [seed]