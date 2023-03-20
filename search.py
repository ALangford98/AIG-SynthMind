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
from util import Queue

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
     #a stack to store the nodes to visit next.using a stack for implementing the breadth-first search 
     # algorithm because it follows the Last-In-First-Out(LIFO)
    
    stack = util.Stack()
    #Adding a the starting state,a list of actions containing nothing, a cost of zero(all  stored in tuple) to the stack we created above, 
    stack.push((problem.getStartState(), [], 0))

    #   # A set with no elements that will store all states visited in a graph tree, as sets do not allow duplicates.
    visited_states = set()

    # a loop runs if the stack is not empty and stops when its empty
    while not stack.isEmpty():
        # remove the action and cost which is on the top of the stack, going to the next node
        state, actions, cost = stack.pop()

        # check if the current state is goal state, if true, we will return the actions taken to get to it
        if problem.isGoalState(state):
            return actions

        # add the current state to the stack of visited states
        visited_states.add(state)

        # expand the node and add its successors to the stack
        for next_state, action_to_successor, cost_to_successor in problem.getSuccessors(state):
            # Check if the next state is already visited or not
            if next_state not in visited_states:
                 # Create a new list of actions with the next action appended to the current list of actions
                next_actions = actions + [action_to_successor]
                # Calculate the new cost by adding the of cost of parent node and cost to get to successor node
                new_cost = cost + cost_to_successor
                 # Add the next state, the new list of actions, and the new cost to the stack  
                stack.push((next_state, next_actions, new_cost))

   # return an empty list if goal state is not found
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

     # A set with no elements that will store all states visited in a graph tree, as sets do not allow duplicates.
    visited_states = set()
    #a queue to store the nodes to visit next.using queue for implementing the breadth-first search 
    # algorithm because it follows the First-In-First-Out (FIFO)
    fringe = util.Queue()
    #a starting node created as tuple with a starting state and a cost of zero
    start_node = (problem.getStartState(), [], 0)
    #We push the node to the que
    fringe.push(start_node)

     #it loops while the queue is not empty, pops a tuple with state, actions, and cost from the queue.
    while not fringe.isEmpty():
        state, actions, cost = fringe.pop()
        # check if the current state is goal state, if true, we will return the actions taken to get to it
        if problem.isGoalState(state):
            return actions
        #if current state is not visited, add it to visited_states
        if state not in visited_states:
            visited_states.add(state)

            # for every unvisited successor of the current state, append the action taken to the list of actions taken to get to the successor 
            # and make an updated node with updated actions, cost and successor state and we finnally add the new node to the fringe que
            for successor, action_to_successor, cost_to_successor in problem.getSuccessors(state):
                # Check if the next state is already visited or not
                if successor not in visited_states:
                     # Create a new list of actions with the next action appended to the current list of actions
                    actions_to_successor = actions + [action_to_successor]
                    # Create a new node with the next state, updated list of actions, and the updated cost
                    new_node = (successor, actions_to_successor, cost + cost_to_successor)
                     # Add the new node to the queue
                    fringe.push(new_node)
    
    # return an empty list if goal state is not found
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    #a priority queue that stores nodes to be expanded in order
    frontier = util.PriorityQueue()
    #we add start start state, cost of zero and list of actions as a tuple to the priority queue
    frontier.push((problem.getStartState(), [], 0), 0)
     # A set with no elements that will store all states visited in a graph tree, as sets do not allow duplicates.
    visited_states = set()

    #if priority queue is not empty, we pop the node with the least cost and 
    # if the current state is the goal state, we return the list of actions taken to this state
    while not frontier.isEmpty():
        state, actions, cost = frontier.pop()
        
        if problem.isGoalState(state):
            return actions
        #if current state is not explored, add to the explored set, get its successor, for all successors we update the list of actions and 
        # total cost to get to that successor, and add it to the frontier
        if state not in visited_states:
            visited_states.add(state)
            for next_state, action_to_successor, cost_to_successor in problem.getSuccessors(state):
                # Create a new list of actions with the next action appended to the current list of actions
                new_actions = actions + [action_to_successor]    
                 # Calculate the new cost by adding the of cost of parent node and cost to get to successor node
                total_cost = cost + cost_to_successor
                 # Add the next state, the new list of actions, and the new cost to the priority queue and total cost as priority
                frontier.push((next_state, new_actions, total_cost), total_cost)
   
    # return an empty list if goal state is not found
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    #starting state of the problem
    start_state = problem.getStartState()
     # A set with no elements that will store all states visited in a graph tree, as sets do not allow duplicates.
    visited_states = set()
    #priority queue that will be used as fringe for search
    queue = util.PriorityQueue()
    #we add start start state, cost of zero and list of actions as a tuple to the priority queue
    queue.push((start_state, [], 0), 0)


    #it loops while the queue is not empty, pops a tuple with state, actions, and cost from the queue.
    while not queue.isEmpty():
        current_state, actions, cost = queue.pop()

        #If the current state is visited, it ontinues to the next iteration.
        if current_state in visited_states:
            continue

        # check if the current state is goal state, if true, we will return the actions taken to get to it
        if problem.isGoalState(current_state):
            return actions
        
        #Otherwise, the current state is added to the set of visited states.
        visited_states.add(current_state)

        #loop to add successors to the queue with new priority  
        for successor, action, cost_to_successor in problem.getSuccessors(current_state):
            if successor not in visited_states:
                # Create a new list of actions with the next action appended to the current list of actions
                new_actions = actions + [action]
                # Calculate the new cost by adding the of cost of parent node and cost to get to successor node
                new_cost = cost + cost_to_successor
                # calculate the priority for the new state by adding the cost to get to that state and its heuristic value
                priority = new_cost + heuristic(successor, problem)
                # add the new state, its action, cost and its priority to the queue
                queue.push((successor, new_actions, new_cost), priority)

    # return an empty list if goal state is not found
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
