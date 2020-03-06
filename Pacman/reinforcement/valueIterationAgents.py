# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        stateList = self.mdp.getStates()
        tempVal = util.Counter()
        for i in range(self.iterations):
          for s in stateList:
            valQ = self.computeQValues(s)
            if len(valQ) == 0:
              tempVal[s] = 0
            else: 
              tempVal[s] = max(valQ)
          for s in stateList:
            self.values[s] = tempVal[s]

    def computeQValues(self, state):
      """
          Returns the list of all Q-values for all actions from the given state.  
      """
      actionList = self.mdp.getPossibleActions(state)
      valQ = []
      for a in actionList:
        valQ += [self.computeQValueFromValues(state, a)]
      return valQ

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        valQ = 0
        transitionList = self.mdp.getTransitionStatesAndProbs(state, action)
        numSuccessorStates = len(transitionList)
        nextStateList = [transitionList[x][0] for x in range(numSuccessorStates)]
        probList = [transitionList[x][1] for x in range(numSuccessorStates)]
        rewardList = [self.mdp.getReward(state, action, nextStateList[x]) for x in range(numSuccessorStates)]
        for index in range(numSuccessorStates):
          valQ += probList[index]*(rewardList[index] + self.discount*self.values[nextStateList[index]])
        return valQ
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
          return None
        actionList = self.mdp.getPossibleActions(state)
        actionDict = util.Counter()
        for a in actionList:
          actionDict[a] = self.getQValue(state, a)
        action = actionDict.argMax()
        return action
        stateValues = self.getValue(state)

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def stateGenerator(self, states):
        while True:
          for s in states:
            yield s

    def runValueIteration(self):
        stateList = self.mdp.getStates()
        gen = self.stateGenerator(stateList)
        for i in range(self.iterations):
          currState = next(gen)
          actionList = self.mdp.getPossibleActions(currState)
          valQ = []
          for a in actionList:
            valQ += [self.computeQValueFromValues(currState, a)]
          if len(valQ) == 0:
            self.values[currState] = 0
          else: 
            self.values[currState] = max(valQ)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        priority = util.PriorityQueue()
        pred = self.getPredecessor()
        stateList = self.mdp.getStates()
        for s in stateList:
          if not self.mdp.isTerminal(s):
            valQ = self.computeQValues(s)
            diff = abs(self.values[s] - max(valQ))
            priority.push(s, -diff)
        for iteration in range(self.iterations):
          if priority.isEmpty():
            break
          s = priority.pop()
          valQ = self.computeQValues(s)
          if not self.mdp.isTerminal(s):
            self.values[s] = max(valQ)
          for p in pred[s]:
            diff = abs(self.values[p] - max(self.computeQValues(p)))
            if diff > self.theta:
              priority.update(p, -diff)

    def getPredecessor(self):
        """
          Returns the predecessors for all states in dictionary format.  (key, value) = (state, list of predecessors)
        """
        stateList = self.mdp.getStates()
        pred = dict()
        for s in stateList:
          actionList = self.mdp.getPossibleActions(s)
          for a in actionList:
            transitionList = self.mdp.getTransitionStatesAndProbs(s, a)
            numSuccessor = len(transitionList)
            nextStates = [transitionList[x][0] for x in range(numSuccessor)]
            for ns, prob in transitionList:
              if prob != 0:
                if ns in pred:
                  pred[ns].add(s)
                else:
                  pred[ns] = set()
                  pred[ns].add(s)
        return pred