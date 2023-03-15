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
import searchAgents

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
    # util.raiseNotDefined()
    # a set that stores nodes that have been explored, no duplicate nodes 
    explored_nodes = set()
    #explored_nodes = set(explored_nodes)

    # fringe will be a stack since strategy for dfs expands the deepest nodes first
    fringe = util.Stack()
    
    # getting the start state   
    initial_state = problem.getStartState()
    # initialising the search node with the state,actions and cost
    start_node = (initial_state,[],0)

    # adding the start node to the fringe
    fringe.push(start_node)
    
    
    while not fringe.isEmpty():
          # if the fringe is not empty then pop the node from the fringe 
        node ,actions,costs= fringe.pop()
       
         # then check if the node popped is the goal state if it is return the actions
        if problem.isGoalState(node):
            return actions
        else:
            # otherwise add it to the explored_nodes list then expand that node 
            if node not in explored_nodes:
                explored_nodes.add(node)
                  # and add its successor to the fringe
                for successor,action,cost in problem.getSuccessors(node):
                     # total actions : a sum of actions that lead to the previous node( parent )and actions that 
                    # lead to the child
                    new_action = actions + [action]
                     # the cumulative cost : a sum of cost of parent node and successor node
                    new_cost = cost + costs
                    new_node = (successor,new_action,new_cost)
                    fringe.push(new_node)

    #if fringe is empty return failure
    return  0     


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
   # util.raiseNotDefined()
    # util.raiseNotDefined()
    # a set that stores nodes that have been explored, no duplicate nodes 
    explored_nodes = set()
    #explored_nodes = set(explored_nodes)

    # fringe will be a queue since strategy for bfs expands the shallowest nodes first
    fringe = util.Queue()
    
    # getting the start state   
    initial_state = problem.getStartState()
    # initialising the search node with the state,actions and cost
    start_node = (initial_state,[],0)

    # adding the start node to the fringe
    fringe.push(start_node)
    
    
    while not fringe.isEmpty():
          # if the fringe is not empty then pop the node from the fringe 
        node ,actions,costs= fringe.pop()
       
         # then check if the node popped is the goal state if it is return the actions
        if problem.isGoalState(node):
            return actions
        else:
            # otherwise add it to the explored_nodes list then expand that node 
            if node not in explored_nodes:
                explored_nodes.add(node)
                  # and add its successor to the fringe
                for successor,action,cost in problem.getSuccessors(node):
                     # total actions : a sum of actions that lead to the previous node( parent )and actions that 
                    # lead to the child
                    new_action = actions + [action]
                     # the cumulative cost : a sum of cost of parent node and successor node
                    new_cost = cost + costs
                    new_node = (successor,new_action,new_cost)
                    fringe.push(new_node)

    #if fringe is empty return failure
    return  0          
           


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
     # a set that stores nodes that have been explored, no duplicate nodes 
    explored_nodes = set()
    #explored_nodes = set(explored_nodes)

    # fringe will be a priority queue since strategy for ucs  expands the node with least cumulative cost first
    fringe = util.PriorityQueue()
    
    # getting the start state   
    initial_state = problem.getStartState()
    # initialising the search node with the state,actions and cost
    start_node = (initial_state,[],0)

    # adding the start node to the fringe
    fringe.push(start_node,0)
    
    while not fringe.isEmpty():
          # if the fringe is not empty then pop the node from the fringe 
        node ,actions,costs= fringe.pop()
       
         # then check if the node popped is the goal state if it is return the actions
        if problem.isGoalState(node):
            return actions
        else:
            # otherwise add it to the explored_nodes list then expand that node 
            if node not in explored_nodes:
                explored_nodes.add(node)
                  # add node successor to the fringe
                for successor,action,cost in problem.getSuccessors(node):
                      # total actions : a sum of actions that lead to the previous node( parent )and actions that 
                    # lead to the child
                    new_action = actions + [action]
                     # the cumulative cost : a sum of cost of parent node and successor node
                    new_cost = cost + costs
                    #creating a new node
                    new_node = (successor,new_action,new_cost)
                    # adding the new node to the fringe
                    fringe.push(new_node,new_cost)

    #if there is no solution(fringe is empty) return failure
    return  0          
           

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    #   # a set that stores nodes that have been visited no duplicate nodes 
    explored_nodes = set()

    # fringe will be a priority queue ordered by f(n) = g(n) + h(n)
    fringe = util.PriorityQueue()
    
    # getting the initial state from problem
    initial_state = problem.getStartState()
    # initialising the search node with the state,actions ,cost and heuristic
    start_node = (initial_state,[],0,heuristic(initial_state,problem))
    
    # evaluation function = path-cost + heuristic 
    f = start_node[2] + start_node[3]

    # adding the start node to the fringe and evaluation function as the priority
    fringe.push(start_node,f)
    
    while not fringe.isEmpty():
          # if the fringe is not empty then pop the node with least f(n) = g(n) + h(n) from the fringe 

        node ,actions,costs,new_heuristic= fringe.pop()
       
         # then check if the node  is the goal state if it is return the actions
        if problem.isGoalState(node):
            return actions
        else:
            # if node is not goal then add it to the explored_nodes list then expand that node 
            if node not in explored_nodes:
                explored_nodes.add(node)
                  # expand and add its successor to the fringe
                for successor,action,cost in problem.getSuccessors(node):
                    # total actions : a sum of actions that lead to the previous node( parent )and actions that 
                    # lead to the child
                    new_action = actions + [action]
                    # the cumulative cost : a sum of cost of parent node and successor node
                    new_cost = cost + costs
                    # the heuristic  of the successor node
                    new_heuristic = heuristic(successor,problem)
                    # creating a new node 
                    new_node = (successor,new_action,new_cost,new_heuristic)
                    # the evaluation function : a sum of new_cost and new_heuristic 
                    evaluation_function = new_cost + new_heuristic
                    # adding the new node to the fringe with the evaluation function as priority
                    fringe.update(new_node,evaluation_function)

    #if there is no solution(fringe is empty) return 0
    return  0          
           






# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

