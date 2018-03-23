import numpy as np
from BlackjackSM import BlackjackSM
from collections import defaultdict
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# build network
NUM_LAYERS = 11
k = 13
# k = np.rint(NUM_LAYERS / 2 + 0.5)

state_machine = BlackjackSM()
n_in = state_machine.len_state
n_out = state_machine.len_actions

layers_size = {-1: n_in}
factor = (n_out/k/n_in)**(1/(NUM_LAYERS - 1))
for layer in range(NUM_LAYERS):
	layers_size[layer] = int(np.rint(k*n_in * factor**(layer)))
print(layers_size)

modules = []
for i in layers_size.keys():
	if i == -1: continue
	modules.append(nn.Linear(layers_size[i-1],layers_size[i]))
	if i < NUM_LAYERS - 1:
		modules.append(nn.BatchNorm1d(layers_size[i]))
		modules.append(nn.ReLU())
		# modules.append(nn.Dropout(0.15))

# define model
class DQN(nn.Module):
	def __init__(self, modules):
		super(DQN, self).__init__()
		for layer, module in enumerate(modules):
			self.add_module("layer_" + str(layer), module)

	def forward(self, x):
		for layer in self.children():
			x = layer(x)
		return x

# initalize model
model = DQN(modules)
# model.load_state_dict(torch.load("models/blackjack_DQN_" + str(NUM_LAYERS) + ".pt"))
try: 
	model.load_state_dict(torch.load("models/blackjack_DQN_" + str(NUM_LAYERS) + "-" + str(k) + ".pt"))
	print("loaded saved model")
except:
	print("no saved model")
	exit()
model.eval()

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
		scores = model(Variable(FloatTensor(np.array([state])), volatile=True)).data
		mask = ByteTensor(1 - state_machine.mask_for(state))
		best_action = (scores.masked_fill_(mask, -16)).max(-1)[1][0]
		policy[tuple(self_state)].append(best_action)
		# if state[0] == 11:
		# 	pdb.set_trace()

act_lookup = {0: "hit", 1: "stand", 2: "surrender", 3: "double", 4: "split"}
with open("policies/blackjack_DQN_pol_" + str(NUM_LAYERS) + "-" + str(k) + ".csv", "w") as file:
	file.write("state,")
	for dealer_hand in range(2,11):
		file.write(str(dealer_hand) + ",")
	file.write("11\n")
	for self_state, actions in policy.items():
		file.write(str(self_state[0]) + " " + str(self_state[1]) + " " + str(self_state[3]) + ",")
		for action in actions[:-1]:
			file.write(str(act_lookup[action]) + ",")
		file.write(str(act_lookup[actions[-1]]) + "\n")