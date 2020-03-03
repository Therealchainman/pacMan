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
    visiting = util.Stack()
    actions = util.Stack()
    visited = set()
    directions = []
    node = problem.getStartState()
    visiting.push([node])
    if problem.isGoalState(node):
        return directions
    while not visiting.isEmpty():
        # the first path that I added to my queue. 
        path = visiting.pop()
        # if I have actions, I need to take my most recent actions that corresponds to the same path
        if not actions.isEmpty():
            directions = actions.pop()
            action = directions[-1]
        # Getting the last node in my path, most recent node I'v visited. 
        node = path[-1]
        visited.add(node)
        if problem.isGoalState(node):
            return directions
        # Create a new path for each successor that has not been explored.  So I am exploring all successors.
        # And I am appending it to a new path that is pushed into my queue. 
        for s in problem.getSuccessors(node):
            if s[0] not in visited:
                new_path = list(path)
                new_path.append(s[0])
                visiting.push(new_path)
                new_action = list(directions)
                new_action.append(s[1])
                actions.push(new_action)
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
    visiting = util.Queue()
    actions = util.Queue()
    visited = set()
    directions = []
    node = problem.getStartState()
    visiting.push([node])
    visited.add(node)
    if problem.isGoalState(node):
        return directions
    while not visiting.isEmpty():
        # the first path that I added to my queue. 
        path = visiting.pop()
        # if I have actions, I need to take my most recent actions that corresponds to the same path
        if not actions.isEmpty():
            directions = actions.pop()
            action = directions[-1]
        # Getting the last node in my path, most recent node I'v visited. 
        node = path[-1]
        if problem.isGoalState(node):
            return directions
        # Create a new path for each successor that has not been explored.  So I am exploring all successors.
        # And I am appending it to a new path that is pushed into my queue. 
        for s in problem.getSuccessors(node):
            if s[0] not in visited:
                new_path = list(path)
                new_path.append(s[0])
                visiting.push(new_path)
                visited.add(s[0]) 
                new_action = list(directions)
                new_action.append(s[1])
                actions.push(new_action)
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    visiting = util.PriorityQueue()
    actions = util.PriorityQueue()
    costq = util.PriorityQueue()
    visited = set()
    directions = []
    node = problem.getStartState()
    visiting.push([node], 0)
    costq.push(0, 0)
    visited.add(node)
    if problem.isGoalState(node):
        return directions
    while not visiting.isEmpty():
        # the first path that I added to my queue. 
        path = visiting.pop()
        cost = costq.pop()
        # if I have actions, I need to take my most recent actions that corresponds to the same path
        if not actions.isEmpty():
            directions = actions.pop()
            action = directions[-1]
        # Getting the last node in my path, most recent node I'v visited. 
        node = path[-1]
        if problem.isGoalState(node):
            return directions
        # Create a new path for each successor that has not been explored.  So I am exploring all successors.
        # And I am appending it to a new path that is pushed into my queue. 
        for s in problem.getSuccessors(node):
            # I added a second condition that the only time to add a node that has already been visited is when
            # it is a goal state.  The logic is that we could have a path that already reached the goal but,
            # we are now exploring another path that has actually a less cost, but it won't explore the goal
            # cause it thinks "Oh, I have already explored the goal node", but we want to make sure we are 
            # exploring the goal noded regardless if it has been explored or not. 
            if s[0] not in visited or problem.isGoalState(s[0]):
                new_path = list(path)
                new_path.append(s[0])
                new_cost = cost + s[2]
                visiting.push(new_path, new_cost)
                visited.add(s[0])
                costq.push(new_cost, new_cost)
                new_action = list(directions)
                new_action.append(s[1])
                actions.push(new_action, new_cost)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visiting_order = util.PriorityQueueWithFunction(cost_calculator)
    visited = set()
    actions = []
    node = problem.getStartState()
    visiting_order.push(([node], problem, heuristic, actions, [0]))
    visited.add(node)
    if problem.isGoalState(node):
        return actions
    while not visiting_order.isEmpty():
        # the first path that I added to my queue. 
        quintuple = visiting_order.pop()
        # if I have actions, I need to take my most recent actions that corresponds to the same path
        directions = quintuple[3]
        if len(directions) != 0:
            actions = directions[-1]
        # Getting the last node in my path, most recent node I'v visited.
        path = quintuple[0] 
        node = path[-1]
        costsoFar = quintuple[4]
        cost = costsoFar[-1]
        if problem.isGoalState(node):
            return directions
        # Create a new path for each successor that has not been explored.  So I am exploring all successors.
        # And I am appending it to a new path that is pushed into my queue. 
        for s in problem.getSuccessors(node):
            # I added a second condition that the only time to add a node that has already been visited is when
            # it is a goal state.  The logic is that we could have a path that already reached the goal but,
            # we are now exploring another path that has actually a less cost, but it won't explore the goal
            # cause it thinks "Oh, I have already explored the goal node", but we want to make sure we are 
            # exploring the goal noded regardless if it has been explored or not. 
            if s[0] not in visited or problem.isGoalState(s[0]):
                new_path = list(path)
                new_path.append(s[0])
                new_cost = list(costsoFar)
                new_cost.append(cost + s[2])
                new_directions = list(directions)
                new_directions.append(s[1])
                visiting_order.push((new_path, problem, heuristic, new_directions, new_cost))
                visited.add(s[0])
    util.raiseNotDefined()

def cost_calculator(inputs):
    # input is equal to a tuple that represents the (state, problem, heuristic, action, cost)
    stateList = inputs[0]
    state = stateList[-1]
    problem = inputs[1]
    heuristic = inputs[2]
    action = inputs[3]
    cost = inputs[4][-1]
    total_cost = heuristic(state, problem) + cost
    return total_cost


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
