import numpy as np

class proportional_controller():
    def __init__(self):
        """
        States: 
        Init - Initial Boot State
        Found_Line - Line_in_perception_window
        No_Line - Line_NOT_in_perception_window
        """
        self.state = 'Init'
        self.angle_target_robot = 0
        self.robot_x = 0
        self.robot_y = 0
        self.target_x = 0
        self.target_y = 0
        self.k = 0.5
    def initial_state(self):
        #return no velocity, no angle
        return 0, 0

    def control(self, robot, target):
        if target == (None,None):
            return (0,5)
        else:
            robot_x, robot_y = robot
            target_x, target_y = target
            self.calculate_angle(robot_x, robot_y, target_x, target_y)
            return (0,self.angle_target_robot*self.k)

    def calculate_angle(self,robot_x, robot_y, target_x, target_y):
      
        # print(target_x,target_y,robot_x,robot_y)
        angle_world = np.arctan2( int(target_y - robot_y), int(target_x - robot_x) )
        
        angle_robot = angle_world + 90
        if angle_world < 0:
            self.angle_target_robot = angle_robot
        else: self.angle_target_robot = angle_robot*-1
