import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import cv2
import math

from Node import Node
from PriorityQueue import PriorityQueue
from controller import proportional_controller
from robot import robot

debug = 'none'

class world():
    def __init__(self, env_map_name):
        self.env_map_name = env_map_name

        self.env_map = None # Used for rendering
        self.env_map_prev = None
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
        self.env_map_prev = copy.deepcopy(self.env_map)

        self.xlim = self.env_map.shape[0]
        self.ylim = self.env_map.shape[1]

        self.bare_env_map = copy.deepcopy(env_map)

        self.inflated_env_map = inflated_map
        # plt.imshow(self.inflated_env_map)
        # plt.title('inflated')
        # plt.show()
        #raise 'die'


 
    def render(self):
        #env is 500x500x4

        # add a block for each agent with a block of a certain size:
        for robot in self.agents:
            self.env_map = change_pixel_color(self.env_map, robot.x, robot.y, 255,0,0)
            #print(roboot.y-block_size:robot.y+block_size, 3] = 255

        # Here's how I pick out the active pixels
        red_pixs = np.argwhere(cv2.inRange(self.env_map, (50,0,0,0), (255,0,0,255)))
        dead_pixs = np.argwhere(cv2.inRange(self.env_map, (1,0,0,0), (50,0,0,255)))

        # This is how i decay pixels:
        for x,y in red_pixs: 
            self.env_map[x,y,0] *= 0.9
        for x,y in dead_pixs:
            self.env_map = change_pixel_color(self.env_map,x,y,255,255,255)

        if self.curr_step%20 == 0:
            plt.close('all')
            plt.figure()
        # Render the map

        black_pixs = np.argwhere(cv2.inRange(self.env_map_prev, (0,0,0,0), (100,100,100,255)))
        for x,y in black_pixs: 
            self.env_map= change_pixel_color_direct (self.env_map,x,y,0,0,0,255) # with 0 inflation



        plt.ion()
        plt.imshow(self.env_map,animated=True)
        # Update the map in a non-blocking way
        plt.title(f'step {self.curr_step}')
        #self.env_map = copy.deepcopy(self.env_map_prev)


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

    def edit_agent(self, velocity, turn_angle,turn_direction='CW', robot_num = 0):
        """
        Parameters
            velocity: control velocity directly
            turn_angle: angle to turn at (deg)
            turn_direction: either CW or CCW
            robot_num: each robot is implicitly assigned a number on creation
        Returns
            None
        """
        robot = self.agents[robot_num]
        #robot.increase_velocity(velocity)
        robot.set_velocity(velocity)
        if(turn_direction == 'CW'):
            robot.rotate_right(turn_angle)
        else:
            robot.rotate_left(turn_angle)

    def step(self):
        """
        Takes a step and updates the location of agents
        Parameters
            None
        Returns
            Done: True or False if its done or not
                    True if done. False if not done
        """

        # Update the robot
        for robot in self.agents:
            # add randomness to simulate real world:
            velocity_offset = np.random.randint(0,1) 
            #self.x = self.velocity + velocity_offset
            realized_velocity = robot.velocity + velocity_offset


             
            """
            Ok, this is confusing, but there implicitly exists two frames of references
            The Global Coordinate Frame is Origined at top left 
            The Robot Coordinate Frame is originated at the robot
            The Rotation matrix is implicit at 90 degrees 
                so we can apply a coordinate transformation
            So suddenly, we can now no longer need to do any complicated trig 
            """

            theta = robot.angle
            # robot is multiplied by a random value at a different angle. 
            # This is derived from the unit circle
            x = realized_velocity * np.cos( np.deg2rad(theta) )
            y = realized_velocity * np.sin( np.deg2rad(theta) )

            # Apply the rotation
            rotation_matrix = np.array(([0,-1],[1,0]))
            robot_vector = np.array(([x],[y]))
            coordinate_vector = rotation_matrix@robot_vector
            x = coordinate_vector[0]
            y = coordinate_vector[1]
            
            #Check if it impacted a wall

            #Sadly, because we have sine/cosines we have to worry about divide by zero erros
            #But gladly, we can fix this with the engineering trick called the fudge trick
            #Which is just add a random offset:
            x += 1e-3
            y += 1e-3

            #Slopes help to detect where walls are to extrapolate the wall locations
            #This is also known as the Brensenham Line Algorithm
            #Its a super duper dumb way, but it *technically* is part of it
            slope = int(y[0]/x[0])
            slopeinv = int(x[0]/y[0])

            boundary = 5
            # this is a horible way to do this, but I do not really have a better idea
            # it checks for boundaries in both X and Y directions
            # we do this for X and Y to check both horizontal and vertical lines
            for xi in range(int(np.abs(x[0]))):
                xpix = int(robot.x + xi)
                ypix = int(robot.y + (slope) * xi)

                #rpix = self.env_map[xpix-2:xpix+2,ypix-2:ypix+2,0]
                rpix = list(self.bare_env_map[xpix-boundary:xpix+boundary, \
                        ypix-boundary:ypix+boundary,0].flatten())
                gpix = list(self.bare_env_map[xpix-boundary:xpix+boundary, \
                        ypix-boundary:ypix+boundary,1].flatten())
                bpix = list(self.bare_env_map[xpix-boundary:xpix+boundary, \
                        ypix-boundary:ypix+boundary,2].flatten())
                for i in range(len(rpix)):
                    if(rpix[i] < 200 and gpix[i] <200 and bpix[i] <200):
                        self.done = True
                        raise Exception('Beep Boop: I have hurt myself on a wall! Ouch!')
                if (debug == 'visual'):
                    self.env_map[xpix,ypix,0] = 0
                    self.env_map[xpix,ypix,1] = 0
                    self.env_map[xpix,ypix,2] = 255
                    self.env_map[xpix,ypix,3] = 255

            for yi in range(int(np.abs(y[0]))):
                ypix = int(robot.y + yi)
                xpix = int(robot.x + (slopeinv) * yi)

                #rpix = self.env_map[xpix-2:xpix+2,ypix-2:ypix+2,0]
                rpix = list(self.bare_env_map[xpix-boundary:xpix+boundary, \
                        ypix-boundary:ypix+boundary,0].flatten())
                gpix = list(self.bare_env_map[xpix-boundary:xpix+boundary, \
                        ypix-boundary:ypix+boundary,1].flatten())
                bpix = list(self.bare_env_map[xpix-boundary:xpix+boundary, \
                        ypix-boundary:ypix+boundary,2].flatten())
                for i in range(len(rpix)):
                    if(rpix[i] < 200 and gpix[i] <200 and bpix[i] <200):
                        self.done = True
                        raise Exception('Beep Boop: I have hurt myself on a wall! Ouch!')
                if (debug == 'visual'):
                    self.env_map[xpix,ypix,0] = 0
                    self.env_map[xpix,ypix,1] = 0
                    self.env_map[xpix,ypix,2] = 255
                    self.env_map[xpix,ypix,3] = 255

            #update locations:
            robot.y += int(y[0])
            robot.x += int(x[0])

            self.check_agent_goal()
            self.curr_step+=1

            


            return(self.done)
    def set_goals(self, goalx, goaly):
        """
        parameters
            goalx: x goal location
            goaly: y goal location
        """
        self.xgoal = goalx # dont ask why these are swapped
        self.ygoal = goaly # if i was smarter i woulnt do this

        boundaryExpansion = self.goal_inflation
        self.env_map[goalx-boundaryExpansion:goalx+boundaryExpansion\
                ,goaly-boundaryExpansion:goaly+boundaryExpansion,0] = 0
        self.env_map[goalx-boundaryExpansion:goalx+boundaryExpansion\
                ,goaly-boundaryExpansion:goaly+boundaryExpansion,1] = 255
        self.env_map[goalx-boundaryExpansion:goalx+boundaryExpansion\
                ,goaly-boundaryExpansion:goaly+boundaryExpansion,2] = 0
        self.env_map[goalx-boundaryExpansion:goalx+boundaryExpansion\
                ,goaly-boundaryExpansion:goaly+boundaryExpansion,3] = 255
        # also set pixels for the smaller environmental map
        self.bare_env_map = change_pixel_color(self.bare_env_map, goalx, goaly,0,255,0, self.goal_inflation)



    def check_agent_goal(self, state = None):
        """
        This should check if agent is near enough to goal
        """
        if state != None:
            boundaryExpansion = self.goal_inflation # dont ask why i did this either.
            realx = state[0]
            realy = state[1]

            if( self.xgoal - boundaryExpansion < realx < self.xgoal + boundaryExpansion\
                and self.ygoal - boundaryExpansion < realy < self.ygoal + boundaryExpansion):
                print('I have reached goal')
                return True
            else: return False

        boundaryExpansion = self.goal_inflation # dont ask why i did this either.
        for agent in self.agents:
            realx = agent.x
            realy = agent.y

            if( self.xgoal - boundaryExpansion < realx < self.xgoal + boundaryExpansion\
                and self.ygoal - boundaryExpansion < realy < self.ygoal + boundaryExpansion):
                print('I have reached goal')
                self.done = True
    
    def getPossibleMoves(self, state):
        # interact with map to get this

        x = state[0]
        y = state[1]

        

        possible_moves = [(x+1,y+1),
                        (x+1,y-1),
                        (x+1,y),
                        (x-1,y+1),
                        (x-1,y-1),
                        (x-1,y),
                        (x,y+1),
                        (x,y-1)]

        _map = self.inflated_env_map
        moves = []
        for move in possible_moves:
            if move[0] < 500 and move[1] < 500 and move[0] > 0 and move[1] > 0 and _map[move[0], move[1], 2] != 0:
                moves.append(move)
        return moves

    def search(self, agent : robot):
        # Position/Parent/Action/Depth
        # state is the coordinate (x,y)

        initNode = Node((agent.x_start, agent.y_start), None, None, 0)
        frontier = PriorityQueue()
        frontier.insert(0, initNode)
        visited = set()
        while (not frontier.isEmpty()):
            current = frontier.remove()
            if (self.check_agent_goal(current.state)):
                return current
            if (current.state not in visited):
                visited.add(current.state)
                moves = self.getPossibleMoves(current.state)
                for move in moves:
                    nextState = move
                    if (nextState not in visited):
                        nextNode = Node(nextState, current, None, current.depth + 1)
                        frontier.insert(current.depth + 1 + self.heuristic(nextNode.state), nextNode)
        return None
    
    def heuristic(self, state):
        return math.sqrt( (state[0] - self.xgoal)**2 + (state[1] - self.ygoal)**2 )

    def path_to(self, node):
        """Given a goal node, trace back through the parent pointers to
        return an ordered list of Nodes along a path from start to goal."""

        path = []
        currentNode = node
        while currentNode:
            path.append(currentNode.state)
            currentNode = currentNode.parent
        path.reverse()
        return path


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
    my_world.set_goals(goalx, goaly) #random goal is like 183, 304
    if (debug == 'text'):
        print(my_world.xlim,my_world.ylim) 

    agent = my_world.agents[0]
    goalNode = my_world.search(agent)
    path = my_world.path_to(goalNode)
    for coordinate in path:
        x = coordinate[0]
        y = coordinate[1]
        my_world.env_map = change_pixel_color(my_world.env_map, x,y,0,0,255,1)
    #print(type(my_world.env_map))
    
    progress = 0
    i = 0
    error_history = []
    pa = False
    while not my_world.done:
        flag = my_world.step()

        robot = agent.get_position()
        # meanx, meany, start,end= agent.get_target_point_on_line(path)
        meanx, meany, start,end= agent.get_target_point_in_window(path)
        # print(start,end)
        #my_world.env_map=change_pixel_color(my_world.env_map,start[0],start[1], 0,255,0)
        #my_world.env_map=change_pixel_color(my_world.env_map,end[0],end[1], 0,255,0)
        #if meanx != None and meany != None:
            #my_world.env_map=change_pixel_color(my_world.env_map,int(meanx),int(meany), 255,255,0)
        #controller = proportional_controller()
        #velocity, angle = controller.control(robot,(meanx,meany))

        #print(velocity,angle,my_world.agents[0].get_angle())

        pp, perception_window, common_points = get_mean_points(path, None, agent, my_world)
        if( common_points != None ):
            for point in common_points:
                path.remove(point)
        if ( pp != None):
            old_pixs = np.argwhere(cv2.inRange(my_world.env_map, (0,255,255,0), (0,255,255,255)))
            for poi in old_pixs:
                my_world.env_map = change_pixel_color_direct\
                        (my_world.env_map, int(poi[0]),int(poi[1]), 255,255,255,255)
            for poi in perception_window:
                my_world.env_map = change_pixel_color_direct\
                        (my_world.env_map, int(poi[0]),int(poi[1]), 0,255,255,255)
            #for poi in common_points:
                #my_world.env_map = change_pixel_color(my_world.env_map, int(poi[0]),int(poi[1]), 255,0,255)


        fla = False
        k_error = 0

        if( pp == None ):
            velocity = 0
            angle = -5
            k_error = angle * 1
            fla = False
        else:

            my_world.env_map = change_pixel_color(my_world.env_map, int(pp[0]),int(pp[1]), 255,0,255)
            dy = robot[0] - pp[0]
            dx = pp[1] - robot[1]
            rotation = np.rad2deg(np.arctan2 (dy,dx ))

            error = rotation - agent.angle
            error_history.append(error)

            if (error > 180):
                error -= 360
            elif (error<-180):
                error += 360
            

            velocity = 5
            k_error = error*0.2
            fla = True

        if(debug == 'text'):
            print(f'point analysis pp: {pp} robot: {robot}')
            print(f'agents angle {agent.angle}, and the rotation is {rotation}')
            print(f'Point {fla}')
            print(f'We have error {rotation}-{agent.angle} = {error}')
        if(k_error>0):
            #print(f'rotating Left with {fla}')
            my_world.edit_agent(velocity,np.abs(k_error), 'CCW') #default CW
            if(debug == 'text'):
                print(f'now rotating CCW to new angle: {agent.angle}')
        else:
            #print(f'rotating Right with {fla}')
            my_world.edit_agent(velocity,np.abs(k_error), 'CW') #default CW
            if(debug == 'text'):
                print(f'now rotating CW to new angle: {agent.angle}')

        
        if(pa == False):
            scri = input('wait')
            if(scri == 'run'):
                pa = True


        #for p in pp:
            #p_x = p[0]
            #p_y = p[1]
            #my_world.env_map = change_pixel_color(my_world.env_map,p_x,p_y, 255,255,0)


        my_world.render()
        plt.imsave(f'runtime_images/good_run_map5/world_{my_world.curr_step}.jpg', my_world.env_map)
        plt.pause(0.05)
        #k = input('wait')
        if(flag or my_world.curr_step>steps):
            break

        # i+=1
        # if i > steps:
        #     break

    error_metric = False
    if(error_metric):
        plt.close('all')
        plt.ioff()
        plt.figure()
        plt.scatter([i for i in range(len(error_history))], error_history)
        plt.show()

def get_mean_points(path, robot_point, agent, world):
    perception_points = generate_perception_points( agent.angle, 40, np.array([agent.x, agent.y]))
    #print('path')
    #print(type(path))
    #print(len(path))
    #print(type(path[0]))
    #print('pp')
    #print(type(perception_points))
    #print(len(perception_points))
    #print(type(perception_points[0]))

    common_points = list(set(path).intersection(perception_points))
    if(len (common_points) == 0):
        return None, None, None
    else:
        # Gotta get furthest Node
        point_dists = []
        m_dist = 0
        m_dist_i = None
        for i,p in enumerate(common_points):

            d = np.linalg.norm( arrayify(agent.x, agent.y) - arrayify(p[0], p[1]) )
            dx = (agent.x + p[0])**2
            dy = (agent.y + p[1])**2
            #print(dx**2 - dy**2)
            d = np.sqrt( np.abs(dx+dy) )
            if(d > m_dist):
                m_dist = d
                m_dist_i = (p[0],p[1])
            #point_dists.append(d)

        return( m_dist_i , perception_points, common_points)

        return( common_points [np.argmax(d)] , perception_points, common_points)

        #return( acc_x/len(common_points), acc_y/len(common_points) )


def generate_perception_points(theta, Radius, robot_point, perception_angle=25):
    x = robot_point[0]
    y = robot_point[1]
        
    points = []
    for i in range(int(theta-perception_angle), int(theta+perception_angle)):
        for R in range(10,Radius):
            #robot_point = angle_point_picker(i, R)
            robot_point = angle_point_picker_R(i,R)
            world_point = point_R_to_W(robot_point, np.array([x,y]))
            points.append( (world_point[0], world_point[1]) )
        
    return(points)
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


def change_pixel_color(the_map,x,y, r, g, b, window = 3):
    if(x>499-window or y>499-window):
        return(the_map)

    the_map[x-window:x+window,y-window:y+window,0] = r
    the_map[x-window:x+window,y-window:y+window,1] = g
    the_map[x-window:x+window,y-window:y+window,2] = b
    the_map[x-window:x+window,y-window:y+window,3] = 255
    return(the_map)
def change_pixel_color_direct(the_map,x,y, r, g, b,a):
    if(x>499 or y>499):
        return(the_map)
    the_map[x,y,0] = r
    the_map[x,y,1] = g
    the_map[x,y,2] = b
    the_map[x,y,3] = a
    return(the_map)

def check_pixel_color(the_map,x,y, r, g, b):
    r_r = the_map[x,y][0]
    r_g = the_map[x,y][1]
    r_b = the_map[x,y][2]
    if(r_r == r and r_g == g and r_b == b):
        return True
    else:
        return False
 
    return the_map


if __name__ == '__main__':
    main()
