# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

#####################################################
# We used this script to find the best value for q2 #
#####################################################
# import numpy as np
# import analysis

# discounts = np.arange(0.01, 1, 0.01)
# noises = np.arange(0.01, 1, 0.01)

# # Changing only the noise parameter
# print("Changing only the noise parameter:")
# best_noise = None
# best_reward = -np.inf
# for noise in noises:
#     answerDiscount, answerNoise = analysis.question2()
#     answerNoise = noise
#     reward = analysis.question2()
#     reward = reward[0]
#     print(f"noise={noise}, reward={reward}")
#     if reward > best_reward:
#         best_noise = noise
#         best_reward = reward
# print(f"Best noise={best_noise}, best reward={best_reward} \n\n\n\n\n\n\n\n\n")

# # Changing only the discount parameter
# print("\nChanging only the discount parameter:")
# best_discount = None
# best_reward = -np.inf
# for discount in discounts:
#     answerDiscount, answerNoise = analysis.question2()
#     answerDiscount = discount
#     reward = analysis.question2()
#     reward = reward[0]
#     print(f"discount={discount}, reward={reward}")
#     if reward > best_reward:
#         best_discount = discount
#         best_reward = reward
# print(f"Best discount={best_discount}, best reward={best_reward}")

# # Changing both parameters together
# print("\nChanging both parameters together:")
# best_discount = None
# best_noise = None
# best_reward = -np.inf
# for discount in discounts:
#     for noise in noises:
#         answerDiscount, answerNoise = analysis.question2()
#         answerDiscount = discount
#         answerNoise = noise
#         reward = analysis.question2()
#         reward = reward[0]
#         print(f"discount={discount}, noise={noise}, reward={reward}")
#         if reward > best_reward:
#             best_discount = discount
#             best_noise = noise
#             best_reward = reward
# print(f"Best discount={best_discount}, best noise={best_noise}, best reward={best_reward}")


def question2():
    answerDiscount = 0.9
    answerNoise = 0.01
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = None
    answerNoise = None
    answerLivingReward = None
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'
    # return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
