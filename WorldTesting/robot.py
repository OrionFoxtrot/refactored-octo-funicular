import math

import numpy as np

class robot():
    """
    ROBOT COORDINATES 
    Due to coordinate transform, I am using Unit Circle Coordinates for ease
        y
        |
        |
        |  /
        | /
        |/) - angle
        -------------x

    The robot has its own coordinate frame because I'm too stupid to have foresight
    """
    def __init__(self, x_start, y_start):

        # Initial Properties
        self.x_start = x_start
        self.y_start = y_start
        
        #initial start point is literally the start point
        self.x = x_start
        self.y = y_start

        # Properties 
        self.velocity = 2
        # Direction will be continuous
        """
        Angle will be reprsented by CCW angle made from east
        """
        self.angle = 0
    # This is used to print stuff normally

    def __str__(self):
        return f"I'm a robot with position ({self.x}, {self.y}) and v,a ({self.velocity}, {self.angle})"
    
    def get_angle(self):
        return self.angle
    
    def get_position(self):
        """
        Basic getter function
        Parameters
            None
        Returns
            global coordinate X, Y
        """
        return(self.x,self.y)
    
    def increase_velocity(self, vel):
        """
        Velocity actuator
        If I was smart, I would have developed this of off accelerations, but I have no foresight
        Parameters:
            vel: the velocity to be set in Units/Step
        Returns:
            None
        """
        self.velocity+=vel
    def set_velocity(self,vel):
        self.velocity = vel
    def set_angle(self, angle):
        self.angle = angle 

    def rotate_left(self,angle=10):
        """
        Rotation for CCW motion
        Parameters:
            angle: default-10, but can be changed. It stpes in 10 units
        Returns:
            None
        """
        self.angle+=angle
        if(self.angle >= 360):
            self.angle-=360

    def rotate_right(self,angle=10):
        """
        Rotation for CW motion
        Parameters:
            angle: default-10, but can be changed. It stpes in 10 units
        Returns:
            None
        """
        self.angle-=angle
        if(self.angle<0):
            self.angle+=360

    # source: https://stackoverflow.com/questions/13652518/efficiently-find-points-inside-a-circle-sector
    
    def areClockwise(self, v1, v2):
        v1_x, v1_y = v1
        v2_x, v2_y = v2
        return -v1_x*v2_y + v1_y*v2_x > 0

    def isWithinRadius(self, v, radiusSquared):
        v_x, v_y = v
        return v_x*v_x + v_y*v_y <= radiusSquared
    def isInsideSector(self, point, center, sectorStart, sectorEnd, radiusSquared):
        point_x, point_y = point
        center_x, center_y = center
        relPoint = (point_x - center_x, point_y - center_y)

        return not self.areClockwise(sectorStart, relPoint) and self.areClockwise(sectorEnd, relPoint) and self.isWithinRadius(relPoint, radiusSquared)
    
    def get_target_point_on_line(self,points: list):
        from scipy.spatial.distance import cdist
        pos_x, pos_y = self.get_position()
        cs = cdist (np.array([[pos_x,pos_y]]), points)
        closest_point_index = np.argmin(cs)

        point_x, point_y = points[closest_point_index]

        return point_x,point_y,(0,0),(0,0)

    
    def get_target_point_in_window(self, points : list):
        """
        Returns: 
            mean points within perception window
        """
        distance = 50
        angle = 20
        pos_x, pos_y = self.get_position()
        robot_angle = self.get_angle() - 90

        # - 90 converts from robot to world coordinates
        start_x = pos_x - distance * math.cos(math.radians(robot_angle - angle))
        start_y = pos_y - distance * math.sin(math.radians(robot_angle - angle))
        start_sector = (int(start_x), int(start_y))

        end_x = pos_x - distance * math.cos(math.radians(robot_angle + angle))
        end_y = pos_y - distance * math.sin(math.radians(robot_angle + angle))
        end_sector = (int(end_x), int(end_y))
        

        r_points = []
        for p in points:
            if (self.isInsideSector(p, self.get_position(), start_sector, end_sector, distance * distance)):
                r_points.append(p)

        if(len(r_points) == 0):
            #reutrn False, if no points are found
            return None,None,start_sector,end_sector
        mean_x, mean_y = 0, 0
        for x, y in r_points:
            mean_x += x
            mean_y += y

        mean_x /= len(r_points)
        mean_y /= len(r_points)


        return (mean_x, mean_y, start_sector, end_sector)
    

    
