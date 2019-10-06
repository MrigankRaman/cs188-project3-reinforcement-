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
        "*** YOUR CODE HERE ***"
        States = self.mdp.getStates()
        #PossibleActions = mdp.getPossibleActions(state)
       # Prob = mdp.getTransitionStatesAndProbs(state, action)
        #Rewards = mdp.getReward(state, action, nextState)
        i = self.iterations
        while i>0:
            y = self.values.copy()
            for state in States:
                if self.mdp.isTerminal(state):
                    continue
                maxValue = -float('inf')
                for possibleAction in self.mdp.getPossibleActions(state):
                    value = 0
                    for item in self.mdp.getTransitionStatesAndProbs(state, possibleAction):
                       
                        value = value+(item[1])*(self.mdp.getReward(state, possibleAction, item[0])+self.discount*y[item[0]])
                        #print(y[item[0]])
                    if value>maxValue:
                        maxValue = value
                self.values[state] = maxValue
            i-=1
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
        "*** YOUR CODE HERE ***"
        value = 0
        for item in self.mdp.getTransitionStatesAndProbs(state, action):
            value = value+(item[1])*(self.mdp.getReward(state, action, item[0])+self.discount*self.values[item[0]])
        return value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None
        maxValue = -float('inf')
        possibleActions = self.mdp.getPossibleActions(state)
        bestAction = possibleActions[0]
        x = util.Counter()
        for possibleAction in possibleActions:
            value = 0
            for item in self.mdp.getTransitionStatesAndProbs(state, possibleAction):
                value = value+(item[1])*(self.mdp.getReward(state, possibleAction, item[0])+self.discount*self.values[item[0]])
            if value>maxValue:
                maxValue = value
                bestAction = possibleAction
        return bestAction
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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
    
        States = self.mdp.getStates()
        for i in range(self.iterations):
            state = States[i%len(States)]
                #y = self.values.copy()
            if self.mdp.isTerminal(state):
                    
                continue
            maxValue = -float('inf')
            for possibleAction in self.mdp.getPossibleActions(state):
                value = 0
                for item in self.mdp.getTransitionStatesAndProbs(state, possibleAction):
                       
                    value = value+(item[1])*(self.mdp.getReward(state, possibleAction, item[0])+self.discount*self.values[item[0]])
                        #print(y[item[0]])
                if value>maxValue:
                    maxValue = value
            self.values[state] = maxValue
            

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
        "*** YOUR CODE HERE ***"
        PQ = util.PriorityQueue()
        States = self.mdp.getStates()
        pred = {new_list: [] for new_list in States} 
        for state in States:
            if self.mdp.isTerminal(state):
                continue
            for possibleAction in self.mdp.getPossibleActions(state):
                for item in self.mdp.getTransitionStatesAndProbs(state, possibleAction):
                    if(state not in pred[item[0]]):
                        pred[item[0]].append(state)
        for state in States:
            if self.mdp.isTerminal(state):
                continue
            value = self.values[state]
            maxQ = -float('inf')
            for action in self.mdp.getPossibleActions(state):
                Qvalue = self.getQValue(state,action)
                if Qvalue>maxQ:
                    maxQ = Qvalue
            diff = abs(maxQ - value)
            
            PQ.update(state,-diff)
        for i in range(self.iterations):
            if PQ.isEmpty():
                return
            state = PQ.pop()
            if self.mdp.isTerminal(state):
                continue
            maxValue = -float('inf')
            for possibleAction in self.mdp.getPossibleActions(state):
                value = 0
                for item in self.mdp.getTransitionStatesAndProbs(state, possibleAction):
                       
                    value = value+(item[1])*(self.mdp.getReward(state, possibleAction, item[0])+self.discount*self.values[item[0]])
                        #print(y[item[0]])
                if value>maxValue:
                    maxValue = value
            self.values[state] = maxValue
            for p in pred[state]:
                if self.mdp.isTerminal(p):
                    continue
                maxQ = -float('inf')
                value = self.values[p]
                for action in self.mdp.getPossibleActions(p):
                    Qvalue = self.getQValue(p,action)
                    if Qvalue>maxQ:
                        maxQ = Qvalue
                diff = abs(maxQ-value)
                if(diff>self.theta):
                    PQ.update(p,-diff)

