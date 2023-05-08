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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # TODO: Initialize the needed values - DONE
        # TODO: Implement the value iteration algorithm - DONE
        
        # Initialize oldValues to store the value of each state in the previous iteration
        oldValues = util.Counter()

        # Get all states in the MDP
        states = self.mdp.getStates()

        # Run value iteration for the specified number of iterations
        for i in range(iterations):
            # Iterate over all states in the MDP
            for state in states:
                # If the state is a terminal state, skip the update
                if mdp.isTerminal(state):
                    continue

                # Get all possible actions for the current state
                actions = mdp.getPossibleActions(state)
                # Initialize an empty list to store the qValues for each action
                qValues = []

                # Iterate over all actions in the current state
                for action in actions:
                    # Initialize the qValue for the current action to 0
                    qValue = 0

                    # Iterate over all possible next states and their transition probabilities
                    for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                        # Get the reward for transitioning from the current state to the next state under the current action
                        reward = mdp.getReward(state, action, nextState)
                        # Update the qValue using the Bellman equation
                        qValue += prob * (reward + discount * oldValues[nextState])

                    # Append the calculated qValue to the list of qValues for the current state
                    qValues.append(qValue)

                # If there are any actions aavilble (qValues), set the new value of the current state to the maximum qValue
                if qValues:
                    newValue = max(qValues)
                # If there are no actions available (no qValues), set the new value of the current state to 0
                else:
                    newValue = 0

                # Update the value of the current state in the values dictionary
                self.values[state] = newValue

            # Update oldValues to store the values from the current iteration for use in the next iteration
            oldValues = self.values.copy()

    
    
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the qValue of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # TODO: Implement the Bellman equtaion - DONE
        # TODO: Compute the qValue of the given state-action pair using self.values - DONE


        # Initialize the qValue to 0
        qValue = 0

        # Iterate over all possible next states and their transition probabilities
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # Get the reward for transitioning from the current state to the next state under the current action
            reward = self.mdp.getReward(state, action, next_state)
            # Update the qValue using the Bellman equation
            qValue += prob * (reward + self.discount * self.values[next_state])

        # Return the calculated qValue
        return qValue
        
        
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # TODO: Find the action with the highest qValue using the max function - DONE
        # TODO: Compute the best action for the given state using self.values - DONE


        # Get all possible actions for the given state using the MDP's getPossibleActions method
        actions = self.mdp.getPossibleActions(state)

        # If there are no legal actions (i.e., the state is a terminal state), return None
        if not actions:
            return None

        # Initialize an empty list to store the qValues for each action
        qValues = []

        # Iterate over all possible actions
        for action in actions:
            # Calculate the qValue of the current action using the computeQValueFromValues function
            qValue = self.computeQValueFromValues(state, action)
            # Append the (action, qValue) pair to the list of qValues
            qValues.append((action, qValue))

        # generic function to get a qValue from a list because I dont know how lambda works
        def get_qValue(item):
            return item[1]
        
        # Find the action with the highest qValue using the max function
        # Discard the second element of each tuple as it is the actual qValue, not the max value we are looking for
        best_action, _ = max(qValues, key=get_qValue)


        # Return the best action
        return best_action
    

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
