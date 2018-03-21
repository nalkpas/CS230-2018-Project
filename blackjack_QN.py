import numpy as np
import numpy.ma as ma
import random
from BlackjackSM import BlackjackSM
import pdb
from collections import defaultdict

state_machine = BlackjackSM()

EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 2000000

Q = defaultdict(lambda: np.zeros(state_machine.len_actions))
num_episodes = 50000000
decay_period = num_episodes - 500000
gamma = 0.999
alpha = 0.0001

counter = 0
def select_action():
	global counter
	unif_draw = np.random.rand()
	eps = EPS_END + max((EPS_START - EPS_END) * (1 - np.exp((counter - decay_period)/EPS_DECAY)), 0)
	
	scores = Q[state_machine.state()]
	mask = 1 - state_machine.mask()
	masked_scores = ma.masked_array(scores,mask).filled(-16)
	best_action = np.argmax(masked_scores)

	if unif_draw > eps:
		return best_action
	else:
		actions = state_machine.actions()
		actions.remove(best_action)
		return random.choice(actions)

for _ in range(num_episodes):
	state_machine.new_hand()
	state = state_machine.state()

	while not state_machine.terminal:
		action = select_action()
		state_machine.do(action)

		if not state_machine.terminal:
			Q[state][action] += alpha*(gamma*np.max(Q[state_machine.state()]) - Q[state][action])
			state = state_machine.state()
		else:
			Q[state][action] += alpha*(state_machine.reward() - Q[state][action])

def state_to_str(state):
	state = list(state)
	output = ""
	for component in state[:-1]:
		output += str(component) + "-"
	output += str(state[-1])
	return output

with open("models/blackjack_QN.csv","w") as file:
	for state, scores in Q.items():
		file.write(state_to_str(state) + ",")
		for score in scores:
			file.write(str(score) + ",")
		file.write("\n")

