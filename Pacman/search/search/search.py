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
    visited = []
    action_list = []
    node_list = []
    visiting_stack = util.Stack()
    prev_stack = util.Stack()
    node = problem.getStartState()
    visiting_stack.push(node)
    # Base case for the case that you start on the goal state, so there are no actions to take,
    # so it is an empty list. 
    if problem.isGoalState(node):
        return action_list
    while not visiting_stack.isEmpty():
        node = visiting_stack.pop()
        print("nodes visiting:", node)
        if problem.isGoalState(node):
            action_list.pop(0)
            print("amount of actions:", len(action_list))
            print("actions to take:", action_list)
            return action_list
        if node not in visited:
            visited.append(node)
            action_list.append(node)
            print("action list so far:", action_list)
            successor_list = []
            prev_stack.push(node)
            for successor in problem.getSuccessors(node):
                if successor[0] not in visited:
                    successor_list.append(successor[0])
                    visiting_stack.push(successor[0])
            if len(successor_list) == 0:
                prev_stack.pop()
            # The point of this is that we want to see if there are some neighbors available that have not 
            # been explored yet, if that is the case, we are going to put them on stack for visiting. 
            # Okay as we are moving back.  
            # I am stuck, I have no idea how to access this info that is if
            # I go [1,2,3,4,5,6,7], and there is no successor for 7, that we have not visited.
            # What should my code do?  It should do something simple, that is go back to 6, right? 
            # Why does this not work, because everytime I visit something I pop it off my stack.  
            # which is a little troublesome right?  cause I cannot access.  So I need somehow to keep track.
            # instead of popping off, I could keep a backup, that is I am popping off of on list, and then
            # at the end it needs to do something a little bit different.  if len
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
    "*** YOUR CODE HERE ***"
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
