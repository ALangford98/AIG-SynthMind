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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    # Create an empty stack to store the nodes to be explored
    fringe = Stack()

    # Create an empty set to store the explored states
    explored = set()

    # Get the starting state of the problem
    startState = problem.getStartState()

    # Create a node representing the starting state with an empty list of actions
    startNode = (startState, [])

    # Add the starting node to the fringe
    fringe.push(startNode)

    # While there are nodes to explore
    while fringe:
        # Pop the next node from the fringe
        currentState, actions = fringe.pop()

        # If the current state has already been explored, skip to the next node
        if currentState in explored:
            continue

        # Add the current state to the explored list
        explored.add(currentState)

        # If the current state is the goal state, return the list of actions
        if problem.isGoalState(currentState):
            return actions

        # Otherwise, get the successors of the current state and add them to the fringe
        successors = problem.getSuccessors(currentState)

        
        for succState, succAction, succCost in successors:
            # Create a new action sequence by appending the current action to the previous actions
            newAction = actions + [succAction]
            # Create a new node by combining the successor state with the new action sequence
            newNode = (succState, newAction)
            # Add the new node to the fringe to be explored later
            fringe.push(newNode)


    # If there are no more nodes to explore and the goal state has not been found, return None
    return 0



    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    # Create an empty queue to store the nodes to be explored
    fringe = Queue()

    # Create an empty set to store the explored states
    explored = set()

    # Get the starting state of the problem
    startState = problem.getStartState()

    # Create a node representing the starting state with an empty list of actions
    startNode = (startState, [])

    # Add the starting node to the fringe
    fringe.push(startNode)

    # While there are nodes to explore
    while not fringe.isEmpty():
        # Pop the next node from the fringe
        currentState, actions = fringe.pop()

        # If the current state has already been explored, skip to the next node
        if currentState in explored:
            continue

        # Add the current state to the explored list
        explored.add(currentState)

        # If the current state is the goal state, return the list of actions
        if problem.isGoalState(currentState):
            return actions

        # Otherwise, get the successors of the current state and add them to the fringe
        successors = problem.getSuccessors(currentState)

        for succState, succAction, succCost in successors:
            # Create a new action sequence by appending the current action to the previous actions
            newAction = actions + [succAction]
            # Create a new node by combining the successor state with the new action sequence
            newNode = (succState, newAction)
            # Add the new node to the fringe to be explored
            fringe.push(newNode)

    # If there are no more nodes to explore and the goal state has not been found, return None
    return 0


    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    # Create an empty priority queue to store the nodes to be explored
    fringe = PriorityQueue()

    # Create an empty set to store the explored states
    explored = set()

    # Get the starting state of the problem
    startState = problem.getStartState()

    # Create a node representing the starting state with an empty list of actions and a cost of 0
    startNode = (startState, [], 0)

    # Add the starting node to the fringe with a priority of 0
    fringe.update(startNode, 0)

    # While there are nodes to explore
    while not fringe.isEmpty():
        # Pop the node with the lowest priority (i.e. lowest cost) from the fringe
        currentState, actions, cost = fringe.pop()

        # If the current state has already been explored, skip to the next node
        if currentState in explored:
            continue

        # Add the current state to the explored list
        explored.add(currentState)

        # If the current state is the goal state, return the list of actions
        if problem.isGoalState(currentState):
            return actions

        # Otherwise, get the successors of the current state and add them to the fringe with their costs
        successors = problem.getSuccessors(currentState)

        for succState, succAction, succCost in successors:
            newAction = actions + [succAction]
            newCost = cost + succCost
            newNode = (succState, newAction, newCost)
            fringe.push(newNode, newCost)

    # If there are no more nodes to explore and the goal state has not been found, return None
    return 0
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue
    
    # Create an empty priority queue to store the nodes to be explored
    fringe = PriorityQueue()
    
    # Create an empty set to store the explored states
    explored = set()
    
    # Get the starting state of the problem
    startState = problem.getStartState()
    
    # Create a node representing the starting state with an empty list of actions, a cost of 0, and an estimated cost to the goal state
    startNode = (startState, [], 0, heuristic(startState, problem))
    
    # Add the starting node to the fringe with a priority of the estimated total cost
    fringe.push(startNode, startNode[2] + startNode[3])
    
    # While there are nodes to explore
    while not fringe.isEmpty():
        # Pop the node with the lowest priority (i.e. lowest estimated total cost) from the fringe
        currentState, actions, cost, estimatedCost = fringe.pop()
    
        # If the current state has already been explored, skip to the next node
        if currentState in explored:
            continue
        
        # Add the current state to the explored list
        explored.add(currentState)
    
        # If the current state is the goal state, return the list of actions
        if problem.isGoalState(currentState):
            return actions
    
        # Otherwise, get the successors of the current state and add them to the fringe with their costs and estimated costs
        successors = problem.getSuccessors(currentState)
    
        for succState, succAction, succCost in successors:
            newAction = actions + [succAction]
            newCost = cost + succCost
            newEstimatedCost = heuristic(succState, problem)
            newNode = (succState, newAction, newCost, newEstimatedCost)
            fringe.push(newNode, newCost + newEstimatedCost)
    
    # If there are no more nodes to explore and the goal state has not been found, return None
    return 0
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
