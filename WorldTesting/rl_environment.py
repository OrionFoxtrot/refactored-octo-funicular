from collections import deque
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import cv2
import math
import random
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm

from Node import Node
from PriorityQueue import PriorityQueue
from controller import proportional_controller
from robot import robot

numerical_action = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
numerical_action = [-8,-5,-3,-1,0,1,3,5,8]


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
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.5
        self.discount = 1
        self.alpha = 0.1
        self.history = []
        self.model = self._build_model()
        self.memory = deque(maxlen=65536)

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
        self.goal_inflation = 10

        #done flag
        self.done = False

        
    def generate_world(self, env_map_name = None):
        if (env_map_name == None):
            env_map_name = self.env_map_name
        #env_map = Image.open(env_map_name) # Open up the environment
        env_map = cv2.imread(env_map_name,-1)
        #plt.imshow(env_map)
        #plt.show()

        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((5,5),np.uint8)
        env_map = cv2.bitwise_not(env_map)
        env_map = cv2.dilate(env_map, kernel, iterations=1)
        infated_map = cv2.dilate(env_map, kernel2, iterations = 1)
        inflated_map = cv2.bitwise_not(infated_map)
        env_map = cv2.bitwise_not(env_map)

        env_map = np.array(env_map) # Load the map as an array
        self.env_map = env_map
        self.env_map_prev = copy.deepcopy(self.env_map)

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
        robot.increase_velocity(velocity)
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
                        return self.done
                        # raise Exception('Beep Boop: I have hurt myself on a wall! Ouch!')
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
                        return True
                        # raise Exception('Beep Boop: I have hurt myself on a wall! Ouch!')
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
                # print('I have reached goal')
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
    
    def _build_model(self):
        """
        Construct a neural network model using keras. 

        We need outputs to be both negative and positive, so use a linear
        activation function.
        """
        model = Sequential()
        model.add(Dense(10, input_dim=2, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(numerical_action), activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.01))
        model.summary()
        return model
    
    def replay(self, batch_size, iterations):
        """
        Selects a random sample of experiences from the memory to
        train on using batch updates.
        """
        for i in range(iterations):
            minibatch = random.sample(self.memory, batch_size)
            states = np.asarray([state[0] for state, action, reward, \
                                 next_state, done in minibatch])
            nextStates = np.asarray([next_state[0] for state, action, \
                                     reward, next_state, done in minibatch])
            rewards = np.asarray([reward for state, action, reward, \
                                  next_state, done in minibatch])
            actions = np.asarray([action for state, action, reward, \
                                  next_state, done in minibatch])
            notdone = np.asarray([not(done) for state, action, reward, \
                                  next_state, done in minibatch]).astype(int)
            nextVals = np.amax(self.model.predict(nextStates,verbose=0), axis=1)
            targets =  rewards + (nextVals * notdone * self.discount)
            targetFs = self.model.predict(states,verbose=0)
            for i in range(len(minibatch)):
                targetFs[i, actions[i]] = targets[i]
            self.model.fit(states, targetFs, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the given experience in the memory.
        """ 
        self.memory.append((state, action, reward, next_state, done))

    def save_weights(self, filename="model.h5"):
        """
        Saves the network weights to a file.
        """
        self.model.save_weights(filename)
            
    def load_weights(self, filename="model.h5"):
        """
        Reloads the network weights from a file.
        """
        self.model.load_weights(filename)
    
   
def reward_assignment(goal_x,goal_y, inflated_map, map_x=250,map_y=250):
    # 500 x 500 pixel board
    q_table = np.ones( (map_x, map_y) , dtype= np.double)
    q_table = q_table*-1
    open_set = deque()
    goal = (goal_x,goal_y)
    q_table[goal] = 0
    open_set.append((goal,0))
    while(open_set):
        (x,y), dist = open_set.popleft()
        dist += 1
        for offset_x, offset_y in [(1,0),(-1,0),(0,1),(0,-1), (1,1),(-1,-1),(-1,1),(1,-1)]:
            reach = x+offset_x, y+offset_y
            if( 0<=reach[0] < q_table.shape[0] and 0<= reach[1]<q_table.shape[1]):
                if(q_table[reach] == -1):
                    q_table[reach] = dist*1
                    open_set.append((reach,dist))

    max_value = np.max(q_table)
    q_table = max_value-q_table
    black_pixs = np.argwhere(cv2.inRange(inflated_map, (0,0,0,0), (100,100,100,255)))
    for x,y in black_pixs: 
        q_table[x,y] = -100


    # print(str(q_table))
    # plt.imshow(q_table)
    # plt.show()
    return(q_table)


def check_action_validity( point, action ):
    #print( point, action )
    return ( check_point_validity( (point[0][0]+action[0], point[0][1]+action[1]) ) )
        
def check_point_validity( point, xlim=100, ylim=100):
    if( point[0]>=xlim or point[1]>=ylim or point[0]<=0 or point[1]<=0 ):
        return False
    return True

def epsilon_greedy_act(state,my_world):
        """
        Given a state, chooses whether to explore or to exploit based
        on the self.epsilon probability.
        """
        if np.random.rand() <= epsilon:
            return epsilon_act(state)
        else:
            return greedy_act(state,my_world)

def epsilon_act(state):
    # while (True):
    action = random.randrange(0,len(numerical_action))
        # if check_action_validity(state, numerical_action[action]) == True:
    return action
def greedy_act(state, my_world):
    """
    Given a state, chooses the action with the best value.
    """
    act_values = my_world.model.predict(state,verbose=0)
    act_values = act_values.flatten()
    # for i in range(len(act_values)):
    #     max_index = np.argmax(act_values)
    #     if check_action_validity(state,numerical_action[max_index])==True:
    #         return max_index
    #     else:
    #         act_values[max_index] = -np.Inf
    # # return epsilon_act(state)
    return np.argmax(act_values)  # returns action

def epsilon_greedy ( point, q_table, state ):
    if ( np.random.rand() < epsilon ):
        # Take Random action
        while( True ) : # Dumb brute force method
            possible_action = numerical_action[np.random.randint(0,len(numerical_action))]
            if (check_action_validity( point, possible_action ) == True):
                #print('random action return')
                return(possible_action)
    else:
         #Take greedy action
         possible_actions = copy.deepcopy(q_table[state,:])
         while( True ):
             numerical_index = np.argmax(possible_actions)
             possible_action = numerical_action[numerical_index]
             if( check_action_validity(point, possible_action) == True):
                 #print('greeedy action return')
                 #print(possible_action)
                 return(possible_action)
             else:
                 possible_actions[numerical_index] = -np.inf



         
    
def value_calculation ( point,state, action, reward_table, q_table, point_state_map):
    # point is x,y 
    # state is q_table_state
    n_action = numerical_action.index(action)

    qsat = q_table[state, n_action]

    next_point = (point[0]+action[0], point[1]+action[1])
    next_state = point_state_map[next_point]

    reward = reward_table[next_point]
    max_q_next_state = np.max(q_table[next_state,:])
    
    global alpha
    global discount
    val =  (1-alpha)*qsat + alpha * (reward+discount*max_q_next_state)
    #print(val)
    #print(point)
    return(val)
        
epsilon = 1
epsilon_decay = 0.99
epsilon_min = 0.3
discount = 0.9
alpha = 0.1

def stupid_mapping_function(xlim=100,ylim=100):
    my_map = np.zeros([xlim*ylim]).astype(np.uint16)
    print(my_map.shape)
    for i in range(xlim*ylim):
        #print(i)
        my_map[i] =  i
    return my_map.reshape([xlim,ylim])

def trace(q_table, my_world,startx,starty,goalx,goaly, point_state_map, loaded_version):
    x = startx
    y = starty 
    path = []
    str_path = []
    goal = (goalx, goaly)
    word_actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    for it in tqdm(range(int(1e3))):
        # state = point_state_map[x,y]
        # numerical_index = np.argmax(q_table[state,:])
        point = (x,y)
        point = np.reshape(point, (1,2))
        numerical_index = greedy_act(point,my_world)
        if(loaded_version == 1):
            numerical_index -= 1
        action = numerical_action[numerical_index]
        if( check_action_validity( point, action ) == False ):
            break
        if(loaded_version == 1):
            numerical_index -= 1
        str_path.append(word_actions[numerical_index-1])
        #print(str(word_actions[numerical_index]))
        #print(str(q_table[state,:]))
        #_ = input('wait')
        # rotate action 90
        #action = action @ np.array ([ [0, 1], [-1,0] ])
        x = x+action[0]
        y = y+action[1]
        coordinate = (x,y)
        path.append( coordinate )
        if (coordinate == goal):
            break

        
    print(str_path, it, path[-1])
    print(goalx, goaly)
    return(path)
def norm_2d(arr):
    norm = np.linalg.norm(arr)
    arr = arr/norm  # normalized matrix
    return arr

def main():
    global epsilon
    global epsilon_decay
    global epsilon_min
    global discount
    global alpha

    filename = sys.argv[1]
    startx = int(sys.argv[2])
    starty = int(sys.argv[3])
    goalx = int(sys.argv[4])
    goaly = int(sys.argv[5])
    steps = int(sys.argv[6])
    load_flag = int(sys.argv[7])
        
    my_world = world(filename) # World Generation, feed it an input map
    my_world.generate_world() # This parses the map 
    my_world.add_agent(startx, starty) # This adds a agent to the middle
    my_world.set_goals(goalx, goaly) #random goal is like 183, 304
    if (debug == 'text'):
        print(my_world.xlim,my_world.ylim) 

    agent = my_world.agents[0]
 

    reward_table = reward_assignment(goalx,goaly, my_world.inflated_env_map)
    # N NE E SE S SW W NW 
    word_actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    q_table = np.zeros( [my_world.xlim*my_world.ylim, 8] )

    point_state_map = stupid_mapping_function()
    #print(q_table.shape)

    
    # training
    point = (agent.x_start, agent.y_start)
    explored_map = np.zeros([my_world.xlim*my_world.ylim]).astype(np.uint16).reshape([250,250])

    from tqdm import tqdm
    episode = 1000
    attempt = 11
    # iterations
    my_world.load_weights(f'weights/map1/try{attempt-1}_episode_100.h5')
    if(load_flag == 0):
        for s in tqdm(range(episode)):
            point = (agent.x_start, agent.y_start)
            agent.x = agent.x_start
            agent.y = agent.y_start
            my_world.step()
            total_reward = 0
            my_world.done = False
            reward = 0
            point = np.reshape(point, (1,2))
            if s % 50 == 0:
                my_world.save_weights(f"weights/map1/try{attempt}_episode_{s}.h5")
            
            for t in range(int(1000)):

                explored_map[point] = explored_map[point]+1

                # state = point_state_map[point]
                
                #Q Table uses State
                action_n = epsilon_greedy_act( point , my_world) 
                action = numerical_action[action_n]
                # print(action)

                my_world.edit_agent(0,action)
                flag = my_world.step()

                # value = value_calculation ( point, state, action, reward_table, q_table ,point_state_map)

                # q_table[ state, numerical_action.index(action) ] = value

                # print(point, action)
                # new_point = (point[0][0]+action[0], point[0][1]+action[1])
                new_point = agent.get_position()
                # reward = reward_table[new_point]
                reward = t
                if flag:
                    reward -= 500
                if my_world.check_agent_goal(new_point):
                    reward += 1000
                done = my_world.check_agent_goal(new_point) or flag
                new_point = np.reshape(new_point, (1,2))
                total_reward += reward
                # print(reward)

                my_world.remember(point, action_n, reward, new_point, done)

                #print(new_point)
                point = new_point
                if done:
                    break
            my_world.history.append(total_reward)
            print(f"Episode: {s}, reward: {total_reward}")
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            else:
                epsilon = 1
            print(epsilon)
            batchSize = 128
            batchIters = 8
            if len(my_world.memory) > batchSize:
                my_world.replay(batchSize,batchIters)
        my_world.save_weights(f'weights/map1/try{attempt}_episode_{episode}.h5')
            
    # else:
        # q_table = pd.read_csv('rl_q_tables/q_table_run_80e6_run.csv').to_numpy()
    my_world.load_weights(f'weights/map1/try{attempt}_episode_{episode}.h5')

    print(epsilon)
    # print(q_table.shape)
    # print(np.max(q_table))
    #plt.imshow(q_table)
    #plt.show()
    #pd.DataFrame(q_table).to_csv('q_table_run_80e6_run.csv')
    #plt.imshow(norm_2d(explored_map))
    #plt.show()
    
    point = (agent.x_start, agent.y_start)
    goal = (my_world.xgoal, my_world.ygoal)
    #def trace(q_table, startx,starty,goalx,goaly, point_state_map):
    # path = trace(q_table, my_world,agent.x_start, agent.y_start, my_world.xgoal, my_world.ygoal, point_state_map, load_flag)
    # print(path)
        
    # for coordinate in path:
    #     x = coordinate[1]
    #     y = coordinate[0]
    #     my_world.env_map = change_pixel_color(my_world.env_map, x,y,0,0,255,1)
    
    #plt.imshow(q_table)
    #plt.show()
    # plt.imshow(my_world.env_map)
    # plt.show()

    #print('qt',q_table)
    #print(q_table.shape)
    #print(np.max(q_table))

    pa= False
    agent.x = agent.x_start
    agent.y = agent.y_start
    my_world.done = False
    my_world.step()
    
    while not my_world.done:

        flag = my_world.step()

        robot = agent.get_position()
        robot = np.reshape(robot, (1,2))
        action = greedy_act(robot,my_world)
        # meanx, meany, start,end= agent.get_target_point_on_line(path)
        # meanx, meany, start,end= agent.get_target_point_in_window(path)
        # print(start,end)
        #my_world.env_map=change_pixel_color(my_world.env_map,start[0],start[1], 0,255,0)
        #my_world.env_map=change_pixel_color(my_world.env_map,end[0],end[1], 0,255,0)
        #if meanx != None and meany != None:
            #my_world.env_map=change_pixel_color(my_world.env_map,int(meanx),int(meany), 255,255,0)
        #controller = proportional_controller()
        #velocity, angle = controller.control(robot,(meanx,meany))

        #print(velocity,angle,my_world.agents[0].get_angle())

        # pp, perception_window, common_points = get_mean_points(path, None, agent, my_world)
        # if ( pp != None):
        #     old_pixs = np.argwhere(cv2.inRange(my_world.env_map, (0,255,255,0), (0,255,255,255)))
        #     for poi in old_pixs:
        #         my_world.env_map = change_pixel_color_direct\
        #                 (my_world.env_map, int(poi[0]),int(poi[1]), 255,255,255,255)
            # for poi in perception_window:
            #     my_world.env_map = change_pixel_color_direct\
            #             (my_world.env_map, int(poi[0]),int(poi[1]), 0,255,255,255)
            #for poi in common_points:
                #my_world.env_map = change_pixel_color(my_world.env_map, int(poi[0]),int(poi[1]), 255,0,255)


        # fla = False
        # k_error = 0

        # if( pp == None ):
        #     velocity = 0
        #     angle = -5
        #     k_error = angle * 1
        #     fla = False
        # else:

        #     my_world.env_map = change_pixel_color(my_world.env_map, int(pp[0]),int(pp[1]), 255,0,255)
        #     dy = robot[0] - pp[0]
        #     dx = pp[1] - robot[1]
        #     rotation = np.rad2deg(np.arctan2 (dy,dx ))

        #     error = rotation - agent.angle

        #     if (error > 180):
        #         error -= 360
        #     elif (error<-180):
        #         error += 360
            

        #     velocity = 0
        #     k_error = error*0.2
        #     fla = True

        # if(debug == 'text'):
        #     print(f'point analysis pp: {pp} robot: {robot}')
        #     print(f'agents angle {agent.angle}, and the rotation is {rotation}')
        #     print(f'Point {fla}')
        #     print(f'We have error {rotation}-{agent.angle} = {error}')
        # if(k_error>0):
        #     #print(f'rotating Left with {fla}')
        #     my_world.edit_agent(velocity,np.abs(k_error), 'CCW') #default CW
        #     if(debug == 'text'):
        #         print(f'now rotating CCW to new angle: {agent.angle}')
        # else:
        #     #print(f'rotating Right with {fla}')
        #     my_world.edit_agent(velocity,np.abs(k_error), 'CW') #default CW
        #     if(debug == 'text'):
        #         print(f'now rotating CW to new angle: {agent.angle}')

        
        # if(pa == False):
        #     scri = input('wait')
        #     if(scri == 'run'):
        #         pa = True


        #for p in pp:
            #p_x = p[0]
            #p_y = p[1]
            #my_world.env_map = change_pixel_color(my_world.env_map,p_x,p_y, 255,255,0)

        my_world.edit_agent(0,numerical_action[action])
        print(robot,numerical_action[action])
        my_world.render()
        plt.imsave(f'runtime_images/world_{my_world.curr_step}.jpg', my_world.env_map)
        plt.pause(0.05)
        #k = input('wait')
        if(flag or my_world.curr_step>steps):
            break

        # i+=1
        # if i > steps:
        #     break

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


def generate_perception_points(theta, Radius, robot_point, perception_angle=15):
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
