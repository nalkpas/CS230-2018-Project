import numpy as np
import numpy.ma as ma
import random
from BlackjackSM import BlackjackSM
from collections import defaultdict
import pdb

state_machine = BlackjackSM()

def str_to_state(str):
	output = str.split(" ")
	output = [float(v) for v in output]
	output = [int(v) for v in output]
	return tuple(output)

Q = defaultdict(lambda: np.zeros(state_machine.len_actions))
with open("models/blackjack_QN.csv") as file:
	Q_file = [line.strip().split(",") for line in file]
for line in Q_file:
	Q[str_to_state(line[0])] = np.array(line[1:-1],dtype=float)

self_space = []
for hand in range(4, 21):
	if hand > 4:
		self_space.append([hand, 0, 1, 0])
	if hand > 12: 
		self_space.append([hand, 1, 1, 0])
	if hand % 2 == 0:
		self_space.append([hand, 0, 1, 1])
	if hand == 12:
		self_space.append([hand, 1, 1, 1])
self_space = sorted(self_space, key=lambda t: (t[3],t[1],t[0]))

policy = defaultdict(list)
for self_state in self_space:
	for dealer_hand in range(2,12):
		state = self_state + [dealer_hand]
		scores = Q[tuple(state)]
		mask = 1 - state_machine.mask_for(state)
		masked_scores = ma.masked_array(scores,mask).filled(-16)
		best_action = np.argmax(masked_scores)

		policy[tuple(self_state)].append(best_action)

act_lookup = {0: "hit", 1: "stand", 2: "surrender", 3: "double", 4: "split"}
with open("policies/blackjack_QN_pol.csv", "w") as file:
	file.write("state,")
	for dealer_hand in range(2,12):
		file.write(str(dealer_hand) + ",")
	file.write("\n")
	for self_state, actions in policy.items():
		file.write(str(self_state[0]) + " " + str(self_state[1]) + " " + str(self_state[3]) + ",")
		for action in actions:
			file.write(str(act_lookup[action]) + ",")
		file.write("\n")

avg_reward = 0
for episode in range(50000):
	state_machine.new_hand()

	while True:
		scores = Q[state_machine.state()]
		mask = 1 - state_machine.mask()
		masked_scores = ma.masked_array(scores,mask).filled(-16)
		best_action = np.argmax(masked_scores)
		state_machine.do(best_action)

		if state_machine.terminal:
			break

	avg_reward += state_machine.reward() / 50000
print(avg_reward) 
