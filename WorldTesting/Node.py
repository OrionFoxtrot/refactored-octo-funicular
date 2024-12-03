########################################
# CS63: Artificial Intelligence, Lab 2
# Spring 2024, Swarthmore College
########################################

class Node:
    """Bookkeeping class for state space search."""
    def __init__(self, state, parent, action, depth):
        """
        state     type depends on problem, for TrafficJam it's a
                  TrafficJam object, for FifteenPuzzle it's a 
                  FifteenPuzzle object
        parent    pointer to the parent node in the graph
        action    type depends on problem, represents action taken to get
                  to this state
        depth     integer that represents the depth of the node in graph
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def __str__(self):
        result = "\nState:\n" + str(self.state)
        result += "\nDepth: " + str(self.depth)
        return result
