<!-- Give detailed instructions for how to run your code here. Be sure to indicate
where all of your data sets are located on the CS system. -->


# Robot Maze Challenge: Exploring A* Search and Reinforcement Learning for Pathfinding

In our project, we compared AI techniques—A* Search and Reinforcement Learning (RL) — for solving a simulated robot maze problem similar to that of Lab 1.

## Relevant Directories & Files

*Note: Bold denotes directories.*

**Maps**: 500 x 500 pixels maps of maze in png format (also contains Maze11 or Maze12 that are 100 x 100)
**Maps250**: 250 x 250 pixels maps of maze in png format
**runtime_images**: runtime images from the latest run
**weights**: weights from the latest RL run

**WorldTesting**: environment files for running
- **rl_q_tables**: sample q_tables

- Node.py - node class from Lab 2 (used for A*)
- PriorityQueue.py - priority queue class from Lab 2 (used for A*)
- controller.py - line following controller 
- environment.py - environment for A* 
- rl_environment.py - environment for RL
- robot.py - robot class
- testing.py - robot testing environment

## Usage

First, navigate to **WorldTesting** directory.

Note the below input values

- map - map to explore
- (startx, starty) - coordinates of the starting position of the robot, must
  be within bounds of the map
- (goalx, goaly) - coordinates of the goal position for the robot to reach,
  must be within bounds of the map
- steps - how many steps to render for 
- option - for RL, 0 for training, 1 for testing

To run A* search:
```
python3 environment.py ../Maps/<map> <startx> <starty> <goalx> <goaly> <steps>
```

To run RL:
```
python3 rl_environment.py ../Maps/<map> <startx> <starty> <goalx> <goaly> <steps> <option>
```
After running above commands, you can press enter to look at the next render frame by frame, or type "run" and press enter to look at the continuous rendering of the full run. A window would pop up to display the rendering of the environment. 

In our trials, we most often used (25, 25) as the starting position and the coordinates 25 pixels away from bottom and right walls as the goal. (e.g. (475, 475) for the 500 x 500 maps)
___________

**Contributors**: Kanyarin Boonkongchuen (kboonk1@swarthmore.edu), Mark Lohatepanont (ml2828@cornell.edu), Rachel Sun (rsun1@swarthmore.edu)
