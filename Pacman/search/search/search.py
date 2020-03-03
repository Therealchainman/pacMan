# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def graph(problem):
    """
    This creates my graph that to be used in the searches.
    It uses a while loop to go through the state list
    """
    initial_state = problem.getStartState()
    state_dict = dict()
    state_list = [initial_state]
    visited = []
    while len(state_list) != 0:
        state = state_list[-1]
        successor_list = []
        actions_list = []
        for successor in problem.getSuccessors(state):
            successor_list.append(successor[0])
        visited.append(state_list.pop())
        count = 0
        while len(successor_list) != 0:
            if count == 0:
                if successor_list[0] not in visited:
                    state_dict[state] = [successor_list[0]]
                    state_list.append(successor_list[0])
                    successor_list.pop(0)
                else: 
                    state_dict[state] = [successor_list[0]]
                    successor_list.pop(0)
            else: 
                if successor_list[0] not in visited: 
                    state_dict[state].append(successor_list[0])
                    state_list.append(successor_list[0])
                    successor_list.pop(0)
                else: 
                    state_dict[state].append(successor_list[0])
                    successor_list.pop(0)
            count += 1
    return state_dict

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # So each of these sets and lists are going to keep track of important data that I need 
    # in the problem.  For example, the first thing is the visited set which 
    # essentially keeps track if I have actually moved to a node and visited it.  
    # action list keeps track of actions that I have taken to get to a new state.
    # prev_list keeps track of the previous nodes that I have visited.  
    # tracker is keeping track if a node has already been expanded, and still has unexplored 
    # successors.
    # Visiting stack is going to give me a stack of nodes that I will visit and it does it in the 
    # correct order.  While the action stack is doing same thing for actions though.  
    # Basically the point of the visiting stack is give me nodes that I will go to next if there was a successor
    # so it is a stack of all successors.  and I remove the successor once I go visit it.  So it keeps track
    # of nodes I still should visit.  
    visited_set = set()
    action_list = []
    prev_list = []
    tracker = set()
    visiting_stack = util.Stack()
    action_stack = util.Stack()
    node = problem.getStartState()
    visiting_stack.push(node)
    prev_list.append(node)
    # This first if statement is checked if we are starting at the goal state, if so return an 
    # empty action list.  
    if problem.isGoalState(node):
        return action_list
    while not visiting_stack.isEmpty():
        # This gives me the most recent element added to previous list.  so that is where I want to visit now. 
        node = prev_list[-1]
        if problem.isGoalState(node):
            return action_list
        if node not in visited_set:
            visited_set.add(node)
            successor_list = []
            # Here I am expanding the node to see its successors, and if for some reason it has more
            # than one successor that is unexplored then I will track that this node so that it will
            # not be expanded again when I backtrack.
            for successor in problem.getSuccessors(node):
                if successor[0] not in visited_set:
                    successor_list.append(successor[0])
                    visiting_stack.push(successor[0])
                    action_stack.push(successor[1])
                if len(successor_list) > 1: 
                    tracker.add(node)
            # So if the length is 0 that indicates there is nowhere to go except to backtrack.  So to backtrack 
            # I want to remove the action that took me to my current place and remove the node I am at.  
            # so now I will return to the previous node in my while loop.  
            if len(successor_list) == 0:
                prev_list.pop()
                action_list.pop()
            # if the length is not 0 then I want to visit that node that is a successor and I will do that
            # by popping off my visiting stack. and adding it to my prev_list
            else:
                node = visiting_stack.pop()
                action = action_stack.pop()
                action_list.append(action)
                prev_list.append(node)
        # I need to check that a node is not in my tracker, because if it is not in my tracker.  
        # If for some reason the node is not in my tracker I need to keep going back, but 
        # if it is in my tracker that means it still has some successors unexplored.  so then you need
        # to explore those nodes. 
        else:
            if node not in tracker:
                prev_list.pop()
                action_list.pop()
            else: 
                node = visiting_stack.pop()
                action = action_stack.pop()
                action_list.append(action)
                prev_list.append(node)
    util.raiseNotDefined()



"""
Here we go I am going to traverse the graph again.  I still have one simple problem,
Yes I can traverse the entire thing but I do not know how to keep track of the path that lead to the end.  
My first step is to created a visited set.  Let's suppose it will contain just states?  yeah 
And I initialize an action list.  This sounds like a good start. 
These could definite come in handy.  
Now the crucial part of my code is going to be my while loop.  I am going to 
use a stack to know that I need to keep iterating until the stack is empty.
okay so I think it does a lot right now. this code.
But does it do what I want it to do?  That is the question.  


"""

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
