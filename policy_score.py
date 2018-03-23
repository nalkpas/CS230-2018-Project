from BlackjackSM import BlackjackSM
from collections import defaultdict
import pdb

num_episodes = 5000000
full_sim = False
network_params = [(11,13)]
# network_params = [(3,1), (3,3), (3,5), (3,7), (3,9), (3,11), (3,13), (3,15),
# 				  (5,7), (7,7), (9,7), (11,7), (13,7),
# 				  (7,13), (11,9), (11,13)]


with open("policies/optimal_policies/scoring_pol.csv", "r") as file:
	scoring_pol = [line.strip().split(",") for line in file]
scoring_pol = [line[1:] for line in scoring_pol]
scoring_pol = scoring_pol[1:]

def get_pol_name(NUM_LAYERS, k):
	return "policies/blackjack_DQN_pol_" + str(NUM_LAYERS) + "-" + str(k) + ".csv"

def score(input_pol, opt_pol):
	score = 0.
	for opt_line, input_line in zip(opt_pol, input_pol):
		for opt_action, input_action in zip(opt_line, input_line):
			if opt_action == "hit":
				if input_action == "hit":
					score += 1
			if opt_action == "stand":
				if input_action == "stand":
					score += 1
			if opt_action == "split":
				if input_action == "split":
					score += 1
			if opt_action == "double":
				if input_action == "double":
					score += 1
				if input_action == "hit":
					score += 0.5
			if opt_action == "DS":
				if input_action == "double":
					score += 1
				if input_action == "stand":
					score += 1
			if opt_action == "XH":
				if input_action == "surrender":
					score += 1
				if input_action == "hit":
					score += 0.5
			if opt_action == "XS":
				if input_action == "surrender":
					score += 1
				if input_action == "stand":
					score += 0.5
	return score

state_machine = BlackjackSM()
act_lookup = {0: "hit", 1: "stand", 2: "surrender", 3: "double", 4: "split"}
code_lookup = {value: key for key,value in act_lookup.items()}
def expectation(pol, num_episodes = 25000):
	avg_reward = 0
	for episode in range(num_episodes):
		state_machine.new_hand()
		while not state_machine.terminal:
			state = state_machine.state()
			state_key = (state[0], state[1], state[3])
			dealer_hand = state[4]
			try:
				action = pol[state_key][dealer_hand - 2]
			except:
				pdb.set_trace()
			state_machine.do(code_lookup[action])

		avg_reward += state_machine.reward() / num_episodes
	return avg_reward

def str_to_state(str):
	output = str.strip().split(" ")
	output = [float(v) for v in output]
	output = [int(v) for v in output]
	return tuple(output)

results = defaultdict(list)
print("started simulating")
for parameters in network_params:
	NUM_LAYERS, k = parameters
	pol_name = get_pol_name(NUM_LAYERS, k)

	with open(pol_name, "r") as file:
		pol_input = [line.strip().split(",") for line in file]
	pol_values = [line[1:] for line in pol_input][1:]
	pol_dict = {str_to_state(line[0]): line[1:] for line in pol_input[1:]}
	
	results[parameters].append(score(pol_values,scoring_pol ))
	results[parameters].append(expectation(pol_dict, num_episodes=num_episodes))
	print("finished " + str(parameters))

if full_sim:
	with open("policies/blackjack_QN_pol.csv", "r") as file:
		pol_input = [line.strip().split(",") for line in file]
	pol_values = [line[1:] for line in pol_input[1:]]
	pol_dict = {str_to_state(line[0]): line[1:] for line in pol_input[1:]}

	results["QN"].append(score(pol_values, scoring_pol))
	results["QN"].append(expectation(pol_dict, num_episodes=num_episodes))
	print("finished QN")

	with open("policies/optimal_policies/opt_pol.csv", "r") as file:
		opt_pol = [line.strip().split(",") for line in file]
	opt_pol_dict = {str_to_state(line[0]): line[1:] for line in opt_pol[1:]}

	results["optimal"].append(340.)
	results["optimal"].append(expectation(opt_pol_dict, num_episodes=num_episodes))
	print("finished optimal")

	with open("scores.csv","w") as file:
		for network, scores in results.items():
			file.write(str(network) + ",")
			for score in scores[:-1]:
				file.write(str(score) + ",")
			file.write(str(scores[-1]) + "\n")

for network, scores in results.items():
	print(str(network) + ": ", end="")
	for score in scores[:-1]:
		print(str(score) + ", ", end="")
	print(str(scores[-1]))

