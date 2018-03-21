import pdb

with open("policies/opt_pol.csv", "r") as file:
	opt_pol = [line.strip().split(",") for line in file]
opt_pol = [line[1:] for line in opt_pol]
opt_pol = opt_pol[1:]

def get_pol_name(NUM_LAYERS, k):
	return "policies/blackjack_DQN_pol_" + str(NUM_LAYERS) + "-" + str(k) + ".csv"

def score(input_pol):
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

network_params = [(3,1), (3,3), (3,5), (3,7), (3,9), (3,11), 
				  (5,7),
				  (7,7)]
results = {}
for parameters in network_params:
	NUM_LAYERS, k = parameters
	pol_name = get_pol_name(NUM_LAYERS, k)

	with open(pol_name, "r") as file:
		input_pol = [line.strip().split(",") for line in file]
	input_pol = [line[1:] for line in input_pol]
	input_pol = input_pol[1:]

	results[parameters] = score(input_pol)

with open("policies/blackjack_QN_pol.csv", "r") as file:
	input_pol = [line.strip().split(",") for line in file]
input_pol = [line[1:] for line in input_pol]
input_pol = input_pol[1:]
results["QN"] = score(input_pol)

for network, score in results.items():
	print(str(network) + ": " + str(score))

