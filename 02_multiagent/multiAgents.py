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
        # getting a list of remaining food
        foodList = newFood.asList()
        # counter to count the number of food in food list
        numberOfFood = 0
        # a list to store  the distances between pacman and each food pellet
        foodDistances = []
        #finding the distance to the closet food
        for foodPos in foodList:
            # a counter that increments by 1 if there is food in foodList
            numberOfFood +=1
            # using manhattan distance between the pacman and  food,to calculate the distance to the clostest food 
            dist = manhattanDistance(newPos,foodPos)
            # adding the distances to the minDistance array
            foodDistances.append(dist)
           # if there is remaining food get the minimum distance
        if numberOfFood > 0:
              # getting the min distance of pacman to food from distances array
              minFoodDist = min(foodDistances)
        else:
              # if no remaining food make the distance 0 
              minFoodDist = 1  
 
     # an empty list to store the distances between pacman and each ghost
        ghostDistances = []
        
        
        # iterate through the ghost state to get their position
        for ghost in newGhostStates:
            # using manhattan distance between the pacman and food , to calculate the distance to the clostest ghost
            dist = manhattanDistance(newPos, ghost.getPosition())
            # adding the distances to the ghostDistance array
            ghostDistances.append(dist)

         # if ghost distance is not empty  
        if ghostDistances:   
            
          if min(newScaredTimes)> 0:
              #if the ghost is scared, the pacman should prioritize it
              minGhostDist = max(ghostDistances) + 1
                  
          else: 
            #avoid ghost if it is not scared
            minGhostDist = min(ghostDistances) 
        else:
            # if there are no more ghost set minGhostDist to 1
            minGhostDist = 1      

        

        # getting the score
        score = successorGameState.getScore()
         
         #If the closest ghost is within one unit of distance from Pacman, the score is decreased by 100
        if minGhostDist <= 1:
            score -=100
        else:
            #  increase the score by 10 divided by the distance to the closest food to encourage pacman to move towards closest food 
            score += 10.0 / (minFoodDist + 1) 

        #if the ghost are scared them prioritize on eating them
        
       
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
        """
        def minimax(state, depth, agentIndex):
            # if we've reached the desired depth or the game is over, return the evaluation function value and no action
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            # Get the legal actions for the current agent
            actions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:
                # If the current agent is the maximizing player, return the action with the highest value
                return max((minimax(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)[0], action) for action in actions)
            else:
                # If the current agent is the minimizing player, return the action with the lowest value
                nextAgentIndex = agentIndex + 1
                 # If the next agent is the first agent, decrement the depth
                if state.getNumAgents() == nextAgentIndex:
                    nextAgentIndex = 0
                    depth -= 1
                 # Evaluate the minimum value and corresponding action from all possible actions   
                return min((minimax(state.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[0], action) for action in actions)

          # Call the minimax function on the initial game state with the specified depth and starting agent
        _, action = minimax(gameState, self.depth, 0)
         # Return the action with the highest value
        return action
      


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
        def alphaBeta(state, depth, agentPos, alpha, beta):
            # Check if the depth limit has been reached or the game is over
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            # Check if the agent is Pacman (max player)
            if agentPos == 0:
                # Set the default values
                maxScore = float("-inf")
                maxAction = None
                # Loop through all the legal actions
                for action in state.getLegalActions(agentPos):
                    successorState = state.generateSuccessor(agentPos, action)
                    # Call the alphaBeta function recursively for the next agent with the updated alpha and beta values
                    score, _ = alphaBeta(successorState, depth, agentPos + 1, alpha, beta)
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
                for action in state.getLegalActions(agentPos):
                    successorState = state.generateSuccessor(agentPos, action)
                    # If it's the last ghost, increase the depth
                    nextagentPos = agentPos + 1
                    nextDepth = depth
                    if nextagentPos == state.getNumAgents():
                        nextagentPos = 0
                        nextDepth += 1
                    # Call the alphaBeta function recursively for the next agent with the updated alpha and beta values
                    score, _ = alphaBeta(successorState, nextDepth, nextagentPos, alpha, beta)
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
        _,action = alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))
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
        def expectimax(state, depth, agent):
            # Base case: either the search has reached the maximum depth or the game has ended
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            # Check if current agent is the maximizing player (Pacman)
            if agent == 0:
                maxVal = float('-inf')
                maxAction = None
                # Loop through legal actions of current agent and find the maximum value and corresponding action
                for action in state.getLegalActions(agent):
                    successor = state.generateSuccessor(agent, action)
                    val, _ = expectimax(successor, depth, agent+1)
                    if val > maxVal:
                        maxVal = val
                        maxAction = action
                return maxVal, maxAction

            # If agent is not the maximizing player, then it is the minimizing player (Ghost)
            else:
                totalVal = 0
                numActions = 0
                nextAgent = agent + 1
                # If all agents have taken their turn, start from the first agent and decrement the depth
                if state.getNumAgents() == nextAgent:
                    nextAgent = 0
                    depth -= 1
                # Loop through legal actions of current agent and calculate the total value
                for action in state.getLegalActions(agent):
                    successor = state.generateSuccessor(agent, action)
                    val, _ = expectimax(successor, depth, nextAgent)
                    totalVal += val
                    numActions += 1
                # Calculate the average value of all legal actions
                avgVal = totalVal / numActions
                return avgVal, None

        # Call the expectimax function with the current game state, maximum depth, and initial agent
        _, action = expectimax(gameState, self.depth, 0)
        # Return the best action determined by the expectimax algorithm
        return action  
       
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      TODO:
      - Make the agent move towards the food -DONE
      - Try to scare the ghosts by eating the pellets -DONE
      - If the ghosts are scared, chase them. distToGhost = min(distsToGhosts) -DONE
      - Avoid them otherwise -DONE

      
        1. The function starts by extracting some information from the currentGameState object, which represents the current state of the game. Specifically, 
        it gets the position of the Pac-Man player (pacmanPos), the layout of the game board (represented as a grid of food pellets, which are True if they exist and False otherwise), 
        a list of the positions of any capsules (power pellets that make ghosts vulnerable when eaten), a list of the states of all the ghosts 
        (including their positions and whether they are currently scared), and a list of the remaining scared times for each ghost (scaredTimes).
    
        2. The function then calculates the distance to the nearest food pellet (distToFood) using the manhattanDistance function, which measures the Manhattan distance 
        (i.e., the sum of the absolute differences of the x- and y-coordinates) between Pac-Man's position and each food pellet, and takes the minimum of these distances. 
        If there are no food pellets left on the board, distToFood is set to 0.
    
        3. The function also calculates the distance to the nearest ghost (distToGhost) using the manhattanDistance function and the list of ghost states. 
        If at least one ghost is currently scared (i.e., has a positive value in its scaredTimer attribute), Pac-Man prioritizes eating that ghost and sets distToGhost to 
        the minimum of the distances to all scared ghosts. Otherwise, Pac-Man avoids ghosts and sets distToGhost to the maximum of the distances to all ghosts.
    
        4. The function then updates the score by adding rewards and penalties based on the values calculated in steps 2 and 3. Specifically, it adds a reward proportional 
        to the inverse of distToFood plus 1, which incentivizes Pac-Man to move towards food pellets. It subtracts a penalty proportional to the inverse of distToGhost plus 1, 
        which penalizes Pac-Man for being too close to a ghost. It adds a reward equal to the number of capsules remaining, which incentivizes Pac-Man to eat capsules. It adds 
        a reward proportional to the total remaining scared time for all ghosts, which incentivizes Pac-Man to chase scared ghosts. And it subtracts a penalty proportional to the 
        number of remaining food pellets, which incentivizes Pac-Man to eat as much food as possible before completing the level.
    
        5. Finally, the function returns the updated score as its output. This score is used by the game engine to determine how well Pac-Man is doing and to guide its decision-making.

    """
    "*** YOUR CODE HERE ***"
    
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghosts]
    
    # Calculate distance to nearest food pellet
    foodList = food.asList()
    if len(foodList) > 0:
        distToFood = min([manhattanDistance(pacmanPos, food) for food in foodList])
    else:
        distToFood = 0

    # Calculate distance to nearest ghost
    distsToGhosts = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    if min(scaredTimes) > 0:
        # If a ghost is scared, prioritize eating it
        distToGhost = min(distsToGhosts)
    else:
        distToGhost = max(distsToGhosts)

    # Add rewards and penalties to the score
    score = currentGameState.getScore()
    score += 1.0 / (distToFood + 1) # Reward moves towards food
    score -= 1.0 / (distToGhost + 1)# Penalty for being close to a ghost
    score += len(capsules) # Reward for eating capsules
    score += 10 * sum(scaredTimes) # Reward for chasing scared ghosts
    score -= 10 * len(foodList) # Penalty for remaining food

    return score

# Abbreviation
better = betterEvaluationFunction
