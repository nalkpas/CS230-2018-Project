# CS230-2018-Project

## BlackjackSM.py 

__BlackjackSM.py__ is a state machine for playing blackjack. It follows a maximal set of casino blackjack rules and features a number of useful support functions for reinforcement learning. 

## blackjack_DQN.py 

__blackjack_DQN.py__ contains PyTorch code for a fully connected linear deep _Q_-network.

## blackjack_eval.py 

__blackjack_eval.py__ converts the models created by __blackjack_DQN.py__ into .csv files containing the derived optimal policies. 

## blackjack_QN.py 

__blackjack_QN.py__ contains code for traditional _Q_-learning, for comparison.

## QN_eval.py

__QN_eval.py__ converts the model created by __blackjack_QN.py__ into a .csv containing the derived optimal policy. 

## policy_score.py

__policy_score.py__ compares my derived policies against the theoretically optimal policy for blackjack, simulates to find the expected reward of the derived and optimal policies in our model, then writes the results to a .csv file. 

## Miscellaneous

The paper folder contains my writeup for this project. The files folder contains various documents collected through the course of the project. 