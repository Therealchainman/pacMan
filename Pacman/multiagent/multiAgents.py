# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        score = successorGameState.getScore()
        pacmanPosition = currentGameState.getPacmanPosition()
        newPacmanPosition = successorGameState.getPacmanPosition()
        food = currentGameState.getFood()
        foodList = food.asList()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        # This is a tuple that contains the distance to the food originally and the position of the food.
        distanceTuple = [(util.manhattanDistance(food, pacmanPosition), food) for food in foodList]
        distancePacmanFood = [distanceTuple[i][0] for i in range(len(distanceTuple))]
        smallestDistance = min(distancePacmanFood)
        index = 0 
        # This is finding the position of the nearest food to the original pacman.
        while index < len(distancePacmanFood):
            if distanceTuple[index][0] == smallestDistance:
                nearestFoodPos = distanceTuple[index][1]
            index += 1
        newDistanceTuple = [(util.manhattanDistance(food, newPacmanPosition), food) for food in newFoodList]
        newDistancePacmanFood = [newDistanceTuple[i][0] for i in range(len(newDistanceTuple))]
        # This is a list of the distances too food now that pacman moved.
        if (len(newDistancePacmanFood) < len(distancePacmanFood)):
            score += 100
        else:
            smallestDistance = min(newDistancePacmanFood)
            index = 0
            while index < len(newDistancePacmanFood):
                if newDistanceTuple[index][0] == smallestDistance:
                    newNearestFoodPos = newDistanceTuple[index][1]
                index += 1
            nearestFoodDist = util.manhattanDistance(nearestFoodPos, pacmanPosition)
            newUpdateNearestFoodDist = util.manhattanDistance(newNearestFoodPos, newPacmanPosition)
            if newUpdateNearestFoodDist < nearestFoodDist:
                score += 100
        # This is considering the fact that pacman may eat a food and so the lengths are different
        # between the old and new.  The else statement checking that you have moved closer to the 
        # nearest food to your position.
        numAgents = successorGameState.getNumAgents()
        agentList = [i for i in range(numAgents)]
        agentList = agentList[1:]
        newGhostPositionList = [successorGameState.getGhostPosition(i) for i in agentList]
        distancePacmanGhost = [util.manhattanDistance(newGhostPositionList[i], newPacmanPosition) 
        for i in range(len(newGhostPositionList))]
        for dist in distancePacmanGhost:
            if dist == 1:
                score -= 10000
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        maxValue = float("-inf")
        bestActionSoFar = Directions.EAST
        actionList = gameState.getLegalActions(0)
        for action in actionList:
            val = self.value(gameState.generateSuccessor(0, action))
            if val > maxValue:
                maxValue = val
                bestActionSoFar = action
        return bestActionSoFar
        util.raiseNotDefined()

    def value(self, state, agentIndex = 0, depth = 0):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        numAgentsZero = state.getNumAgents() - 1
        if agentIndex < numAgentsZero:
            if agentIndex == numAgentsZero - 1:
                depth += 1
            agentIndex += 1
        else:
            agentIndex = 0
        val = self.maxValue(state, agentIndex, depth) if agentIndex == 0 else self.minValue(state, agentIndex, depth)

        return val

    def minValue(self, state, agentIndex, depth):
        v = float('inf')
        actionList = state.getLegalActions(agentIndex)
        successorList = [state.generateSuccessor(agentIndex, action) for action in actionList]
        for successor in successorList:
            v = min(v, self.value(successor, agentIndex, depth))
        return v

    def maxValue(self, state, agentIndex, depth):
        v = float('-inf')
        agentIndex = 0
        actionList = state.getLegalActions(agentIndex)
        successorList = [state.generateSuccessor(agentIndex, action) for action in actionList]
        for successor in successorList:
            v = max(v, self.value(successor, agentIndex, depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float("-inf")
        beta = float("inf")
        maxValue = float("-inf")
        bestActionSoFar = Directions.EAST
        actionList = gameState.getLegalActions(0)
        for action in actionList:
            val = self.value(gameState.generateSuccessor(0, action), 0, 0, alpha, beta)
            alpha = max(alpha, val)
            if beta < alpha:
                break
            if val > maxValue:
                maxValue = val
                bestActionSoFar = action
        return bestActionSoFar
        util.raiseNotDefined()

    def value(self, state, agentIndex, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        numAgentsZero = state.getNumAgents() - 1
        if agentIndex < numAgentsZero:
            if agentIndex == numAgentsZero - 1:
                depth += 1
            agentIndex += 1
        else:
            agentIndex = 0
        val = self.maxValue(state, agentIndex, depth, alpha, beta) if agentIndex == 0 else self.minValue(state, agentIndex, depth, alpha, beta)
        return val

    def minValue(self, state, agentIndex, depth, alpha, beta):
        v = float('inf')
        actionList = state.getLegalActions(agentIndex)
        for action in actionList: 
            successor = state.generateSuccessor(agentIndex, action)
            v = min(v, self.value(successor, agentIndex, depth, alpha, beta))
            beta = min(beta, v)
            if beta < alpha:
                break
        return v

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        v = float('-inf')
        agentIndex = 0
        actionList = state.getLegalActions(agentIndex)
        for action in actionList: 
            successor = state.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successor, agentIndex, depth, alpha, beta))
            alpha = max(alpha, v)
            if beta < alpha:
                break
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        maxValue = float("-inf")
        bestActionSoFar = Directions.EAST
        actionList = gameState.getLegalActions(0)
        for action in actionList:
            val = self.value(gameState.generateSuccessor(0, action))
            if val > maxValue:
                maxValue = val
                bestActionSoFar = action
        return bestActionSoFar
        util.raiseNotDefined()

    def value(self, state, agentIndex = 0, depth = 0):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        numAgentsZero = state.getNumAgents() - 1
        if agentIndex < numAgentsZero:
            if agentIndex == numAgentsZero - 1:
                depth += 1
            agentIndex += 1
        else:
            agentIndex = 0
        val = self.maxValue(state, agentIndex, depth) if agentIndex == 0 else self.expValue(state, agentIndex, depth)

        return val

    def expValue(self, state, agentIndex, depth):
        v = 0
        actionList = state.getLegalActions(agentIndex)
        uniformProb = 1 / len(actionList)
        for action in actionList:
            successor = state.generateSuccessor(agentIndex, action)
            v += uniformProb * self.value(successor, agentIndex, depth)
        return v

    def maxValue(self, state, agentIndex, depth):
        v = float('-inf')
        agentIndex = 0
        actionList = state.getLegalActions(agentIndex)
        for action in actionList:
            successor = state.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successor, agentIndex, depth))
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    There is one more element of the gamestate that is essential
    for evaluating the state of the game.  
    For instance suppose the game state results in a decrease in 
    the relative distance to the closest food pellet.
    That is desirable correct.  So I should
    subtract the manhattan distance.  This seems to be the simplest.
    But how does it know it comparatively reduce the distance compared
    to original state.  Or rather we rate a game state based on the manhattan
    distance.  This makes sense so think about it if you are closer to a food 
    pellet, would you not want to give that a better evalatuation,  so 
    that means it should have a weight of a negative value so that it is 
    more negative evaluated for when you are farther away from a food pellet.
    Now the problem is do I want to consider all the food or just the closest one, and 
    how do you do that in the first place?  
    """
    foodAmount = currentGameState.getNumFood()
    capsuleList = currentGameState.getCapsules()
    capsuleAmount = len(capsuleList)
    score = currentGameState.getScore()
    numGhosts = currentGameState.getNumAgents() - 1
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    distList = [util.manhattanDistance(foodPos, pacmanPos) for foodPos in foodList]
    if len(distList) != 0:
        distToClosestFood = min(distList)
    else:
        distToClosestFood = 0
    evaluate = 4*score - numGhosts - capsuleAmount - 2*foodAmount - 2*distToClosestFood
    return evaluate
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
