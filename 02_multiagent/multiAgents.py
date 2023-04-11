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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # Remove Stop from legal moves
        # legalMoves = [move for move in legalMoves if move != Directions.STOP]
        


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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Calculate the minimum distance to nearest food
        distances_to_food = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if distances_to_food:  # check if there are distances in the list
            distance_to_closest_food = min(distances_to_food)  # get the minimum distance
        else:  # if there are no distances in the list
            distance_to_closest_food = 1  # set the minimum distance to 1

        # Calculate distance to nearest ghost
        distances_to_ghosts = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if distances_to_ghosts:  # check if there are distances in the list
            if min(newScaredTimes) > 0: # If the ghost is scared
                distance_to_closest_ghost = max(distances_to_ghosts) + 1 # Assign a high value and chase after it
            else: # otherwise
                distance_to_closest_ghost = min(distances_to_ghosts)  # get the minimum distance and avoid it
        else:  # if there are no distances in the list
            distance_to_closest_ghost = 1  # set the minimum distance to 1

        # Calculate score
        # Get the current game score
        score = successorGameState.getScore()

        # Add a reward for being close to food (10 divided by distance to closest food + 1)
        score += 10.0 / (distance_to_closest_food + 1)

        # Add a penalty for being close to a ghost (100 divided by distance to closest ghost + 1)
        score -= 100.0 / (distance_to_closest_ghost + 1)

        # Add a penalty for each remaining food
        score -= 10.0 * successorGameState.getNumFood()

        # Add a reward for each remaining scared ghost
        score += 10.0 * sum(newScaredTimes)
        return score

        # return successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"
        # Define the minimax algorithm
        def minimax(state, depth, agentIndex):
            # Check if the depth limit has been reached or the game is over
            if depth == self.depth or state.isWin() or state.isLose():
                # If the game is over or depth limit is reached, return the evaluation of the state
                # and no action
                return self.evaluationFunction(state), None

            # Check if the agent is Pacman (max player)
            if agentIndex == 0:
                # Set the default values
                maxScore = float("-inf")
                maxAction = None
                # Loop through all the legal actions
                for action in state.getLegalActions(agentIndex):
                    # Get the successor state of the action
                    successorState = state.generateSuccessor(agentIndex, action)
                    # Call the minimax function recursively with the successor state, same depth, and next agent
                    score, _ = minimax(successorState, depth, agentIndex + 1)
                    # Update the maximum score and action
                    if score > maxScore:
                        maxScore = score
                        maxAction = action
                return maxScore, maxAction

            # Otherwise the agent is a ghost (min player)
            else:
                # Set the default values
                minScore = float("inf")
                minAction = None
                # Loop through all the legal actions
                for action in state.getLegalActions(agentIndex):
                    # Get the successor state of the action
                    successorState = state.generateSuccessor(agentIndex, action)
                    # If it's the last ghost, increase the depth and switch to Pacman
                    nextAgentIndex = agentIndex + 1
                    nextDepth = depth
                    if nextAgentIndex == state.getNumAgents():
                        nextAgentIndex = 0
                        nextDepth += 1
                    # Call the minimax function recursively with the successor state, next depth, and next agent
                    score, _ = minimax(successorState, nextDepth, nextAgentIndex)
                    # Update the minimum score and action
                    if score < minScore:
                        minScore = score
                        minAction = action
                return minScore, minAction

        # Call the minimax algorithm with the current game state, depth 0, and agent 0
        _, action = minimax(gameState, 0, 0)
        # Return the action that leads to the maximum score
        return action
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Define the alpha-beta algorithm
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Check if the depth limit has been reached or the game is over
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            # Check if the agent is Pacman (max player)
            if agentIndex == 0:
                # Set the default values
                maxScore = float("-inf")
                maxAction = None
                # Loop through all the legal actions
                for action in state.getLegalActions(agentIndex):
                    successorState = state.generateSuccessor(agentIndex, action)
                    # Call the alphaBeta function recursively for the next agent with the updated alpha and beta values
                    score, _ = alphaBeta(successorState, depth, agentIndex + 1, alpha, beta)
                    # Update the maximum score and action
                    if score > maxScore:
                        maxScore = score
                        maxAction = action
                    # Check if the beta value should be updated and prune if necessary
                    if maxScore > beta:
                        return maxScore, maxAction
                    alpha = max(alpha, maxScore)
                return maxScore, maxAction

            # Otherwise the agent is a ghost (min player)
            else:
                # Set the default values
                minScore = float("inf")
                minAction = None
                # Loop through all the legal actions
                for action in state.getLegalActions(agentIndex):
                    successorState = state.generateSuccessor(agentIndex, action)
                    # If it's the last ghost, increase the depth
                    nextAgentIndex = agentIndex + 1
                    nextDepth = depth
                    if nextAgentIndex == state.getNumAgents():
                        nextAgentIndex = 0
                        nextDepth += 1
                    # Call the alphaBeta function recursively for the next agent with the updated alpha and beta values
                    score, _ = alphaBeta(successorState, nextDepth, nextAgentIndex, alpha, beta)
                    # Update the minimum score and action
                    if score < minScore:
                        minScore = score
                        minAction = action
                    # Check if the alpha value should be updated and prune if necessary
                    if minScore < alpha:
                        return minScore, minAction
                    beta = min(beta, minScore)
                return minScore, minAction

        # Call the alphaBeta algorithm with the current game state, depth 0, agent 0, and default alpha/beta values
        _, action = alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))
        return action
        # util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        def expectimax(gamestate, depth, agentIndex):
            # Check if depth is 0 or the game is over
            if depth == 0 or gamestate.isWin() or gamestate.isLose():
                return self.evaluationFunction(gamestate)

            # If agent is Pacman
            if agentIndex == 0:
                return max_value(gamestate, depth, agentIndex)

            # If agent is a ghost
            else:
                return exp_value(gamestate, depth, agentIndex)

        # Define a helper function for maximizing the value
        def max_value(gamestate, depth, agentIndex):
            legalActions = gamestate.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gamestate)
            maxQ = -float("inf")
            for action in legalActions:
                successorState = gamestate.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gamestate.getNumAgents()
                q = expectimax(successorState, depth, nextAgentIndex)
                maxQ = max(maxQ, q)
            return maxQ

        # Define a helper function for calculating the expected value
        def exp_value(gamestate, depth, agentIndex):
            legalActions = gamestate.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gamestate)
            p = 1.0 / len(legalActions)
            expQ = 0.0
            for action in legalActions:
                successorState = gamestate.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gamestate.getNumAgents()
                q = expectimax(successorState, depth, nextAgentIndex)
                expQ += p * q
            return expQ

        # Choose the action with the highest Q value
        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        bestQ = -float("inf")
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            nextAgentIndex = 1
            q = expectimax(successorState, 0, nextAgentIndex)
            if q > bestQ:
                bestQ = q
                bestAction = action
        return bestAction
    

        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Get the current state information
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Calculate the distance to the nearest food pellet
    foodDistances = [manhattanDistance(pacmanPosition, pellet) for pellet in food.asList()]
    nearestFoodDistance = min(foodDistances) if len(foodDistances) > 0 else 0

    # Calculate the distance to the nearest ghost
    ghostDistances = [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates]
    nearestGhostDistance = min(ghostDistances) if len(ghostDistances) > 0 else 0

    # Calculate the total score
    score = currentGameState.getScore()

    # Add rewards and penalties based on game state
    if nearestFoodDistance > 0:
        score += 1.0 / nearestFoodDistance
    if nearestGhostDistance == 0:
        score -= 1000
    if any(scaredTimes):
        score += 500 / (1 + nearestGhostDistance)
    if pacmanPosition == currentGameState.getPacmanPosition():
        score -= 1
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

