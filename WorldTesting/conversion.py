import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import cv2
import math

from Node import Node
from PriorityQueue import PriorityQueue
from robot import robot

debug = 'none'

class world():
    def __init__(self, env_map_name):
        self.env_map_name = env_map_name

        self.env_map = None # Used for rendering
        self.bare_env_map= None # Used for Checking Walls
        self.inflated_env_map = None
        self.agents = [] # Robots 
        self.curr_step = 0 # The current step

        """
        WORLD COORDINATES
        Origin top left +y goes Right, +x goes down

        (0,0)
        -------------->+y
        |
        |
        |
        |
        |
        ↓ +x
        """


        self.xlim = 0 # map boundaries
        self.ylim = 0

        self.xgoal = 0 #goal locations
        self.ygoal = 0
        self.goal_inflation = 20

        #done flag
        self.done = False

        
    def generate_world(self, env_map_name = None):
        if (env_map_name == None):
            env_map_name = self.env_map_name
        #env_map = Image.open(env_map_name) # Open up the environment
        env_map = cv2.imread(env_map_name,-1)

        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((20,20),np.uint8)
        env_map = cv2.bitwise_not(env_map)
        env_map = cv2.dilate(env_map, kernel, iterations=1)
        infated_map = cv2.dilate(env_map, kernel2, iterations = 2)
        inflated_map = cv2.bitwise_not(infated_map)
        env_map = cv2.bitwise_not(env_map)

        env_map = np.array(env_map) # Load the map as an array
        self.env_map = env_map
        self.xlim = self.env_map.shape[0]
        self.ylim = self.env_map.shape[1]

        self.bare_env_map = copy.deepcopy(env_map)

        self.inflated_env_map = inflated_map
        #plt.imshow(self.inflated_env_map)
        #plt.title('inflated')
        #plt.show()
        #raise 'die'


 
    def render(self):
        #env is 500x500x4

        # add a block for each agent with a block of a certain size:
        for robot in self.agents:
            #print(robot.x,robot.y, type(self.env_map))
            block_size = 5
            self.env_map[robot.x-block_size:robot.x+block_size, \
                    robot.y-block_size:robot.y+block_size, 0] = 255
            self.env_map[robot.x-block_size:robot.x+block_size, \
                    robot.y-block_size:robot.y+block_size, 1] = 0
            self.env_map[robot.x-block_size:robot.x+block_size, \
                    robot.y-block_size:robot.y+block_size, 2] = 0
            self.env_map[robot.x-block_size:robot.x+block_size, \
                    robot.y-block_size:robot.y+block_size, 3] = 255


        # Here's how I pick out the active pixels
        red_pixs = np.argwhere(cv2.inRange(self.env_map, (50,0,0,0), (255,0,0,255)))
        dead_pixs = np.argwhere(cv2.inRange(self.env_map, (1,0,0,0), (50,0,0,255)))

        # This is how i decay pixels:
        for x,y in red_pixs: 
            self.env_map[x,y,0] *= 0.9
        for x,y in dead_pixs:
            self.env_map[x,y,0] = 255
            self.env_map[x,y,1] = 255
            self.env_map[x,y,2] = 255
            self.env_map[x,y,3] = 255

        # Render the map
        plt.imshow(self.env_map)
        # Update the map in a non-blocking way
        plt.draw()

    #Creating a new robot with start position x,y
    def add_agent(self, x, y):
        """
        Parameters
            x: start location x
            y: start location y
        return:
            None
        """
        self.agents.append(robot(x,y))

    
def main():
    if(len(sys.argv) == 1):
        print('defaulting to Attempt.png')
        filename = 'Attempt.png'
        startx = 250
        starty = 250
        steps = 10
    if(len(sys.argv) == 2):
        filename = sys.argv[1]
    if(len(sys.argv) == 7):
        filename = sys.argv[1]
        startx = int(sys.argv[2])
        starty = int(sys.argv[3])
        goalx = int(sys.argv[4])
        goaly = int(sys.argv[5])
        steps = int(sys.argv[6])
    if(len(sys.argv)>7):
        print('incorrect usage, correct usage is:')
        print('python3 environment.py filename')
        print('python3 environment.py filename startx starty goalx goaly steps')
        
    my_world = world(filename) # World Generation, feed it an input map
    my_world.generate_world() # This parses the map 
    my_world.add_agent(startx, starty) # This adds a agent to the middle
    if (debug == 'text'):
        print(my_world.xlim,my_world.ylim) 

    # for i in range(steps):
    #     flag = my_world.step()
    #     my_world.edit_agent(0,20)
    #     print(my_world.agents[0],'angle',my_world.agents[0].get_angle())
    #     my_world.render()
    #     plt.pause(0.8)
    #     if(flag):
    #         break

    agent = my_world.agents[0]
    robot_point = np.array([-50,100]) # x+50, y+50

    #robot_point = angle_point_picker_R(45,50)
    
    #try_t0_do_a_sweep
    degree = 12
    Range = 50
    for i in range(degree):
        for R in range(Range):
            #robot_point = angle_point_picker(i, R)
            robot_point = angle_point_picker_R(i,R)
            world_point = point_R_to_W(robot_point, np.array([agent.x,agent.y]))
            my_world.env_map = change_pixel_color(\
                    my_world.env_map,world_point[0], world_point[1], 255,0,255)
        
    my_world.render()
    plt.show()
   

def point_R_to_W(robot_point, robot_location):
    #robot point is numpy array of x,y
    #robot_point is numpy array
    world_point = np.array([robot_location[0]-robot_point[1],robot_location[1]+robot_point[0]])
    return(world_point)
def angle_point_picker_R(theta, R):
    #robot angle point picker
    # theta in deg
    """
    |
    |   R
    |  /
    | /
    |/) Theta
    -----------
    """
    # sin(theta) = Y/R
    y = R * np.sin( np.deg2rad(theta) )
    x = R * np.cos( np.deg2rad(theta) )
    return(arrayify(int(x),int(y)))

def arrayify(x,y):
    return(np.array([x,y]))

def change_pixel_color(the_map,x,y, r, g, b,window = 5):
    the_map[x-window:x+window,y-window:y+window,0] = r
    the_map[x-window:x+window,y-window:y+window,1] = g
    the_map[x-window:x+window,y-window:y+window,2] = b
    the_map[x-window:x+window,y-window:y+window,3] = 255

    return the_map


if __name__ == '__main__':
    main()
