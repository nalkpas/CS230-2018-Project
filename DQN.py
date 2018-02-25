import numpy as np
import random
from BlackjackSM import BlackjackSM
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

class Memory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, transition):
		if len(self.memory) < self.capacity:
			self.memory.append(transition)
		else:
			self.memory[self.position] = transition
			self.position = (self.position + 1) % self.capacity

	def sample(self, batchsize = 10):
		return random.sample(self.memory, batchsize)

	def __len__(self):
		return len(self.memory)

	def __str__(self):
		return str(self.memory)

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.lin1 = nn.Linear(2, 10)
		self.lin2 = nn.Linear(10, 20)
		self.lin3 = nn.Linear(20, 20)
		self.head = nn.Linear(20, 4)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		return self.head(x)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

state_machine = BlackjackSM()
model = DQN()

optimizer = optim.Adam(model.parameters())
memory = Memory(10000)

counter = 0

def select_action(state):
	global counter
	unif_draw = np.random.rand()
	eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * counter / EPS_DECAY)
	counter += 1

	if unif_draw > eps:
		return model(Variable(FloatTensor(state), volatile=True)).data.type(FloatTensor).max(-1)[1]
	else:
		return LongTensor([random.choice(state_machine.get_actions())])

episode_rewards = []
def plot_rewards():
	plt.figure(2)
	plt.clf()
	rewards_t = torch.FloatTensor(episode_rewards)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.plot(rewards_t.numpy())
	# Take 100 episode averages and plot them too
	if len(episode_rewards) >= 100:
		means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())

	plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return	
	sample = memory.sample(BATCH_SIZE)

	batch = list(zip(*sample))

	not_terminal = ByteTensor(tuple(map(lambda s: s is not None, batch[2])))
	not_terminal_states = Variable(torch.cat([s for s in batch[2] if s is not None]), volatile=True)

	state_batch = Variable(torch.cat(batch[0]))
	action_batch = Variable(torch.cat(batch[1]))
	reward_batch = Variable(torch.cat(batch[3]))

	# import pdb
	# pdb.set_trace()

	Q_sa = model(state_batch).gather(1, action_batch.view(-1,1))

	V_s = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
	V_s[not_terminal] = model(not_terminal_states).max(1)[0]

	V_s.volatile = False

	observed_sa = reward_batch + (V_s * GAMMA)

	loss = F.smooth_l1_loss(Q_sa, observed_sa)

	optimizer.zero_grad()
	loss.backward()
	for param in model.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

num_episodes = 5000
for i_episode in range(num_episodes):
	state_machine.new_hand()
	state = FloatTensor([state_machine.state()])

	t = 0
	while True:
		action = select_action(state)
		state_machine.do(int(action[0]))
		reward = FloatTensor([state_machine.get_reward()])
		done = state_machine.terminal 

		if not done:
			next_state = FloatTensor([state_machine.state()])
		else:
			next_state = None

		memory.push([state, action, next_state, reward])
		state = next_state

		optimize_model()
		t = t + 1

		if done:
			episode_rewards.append(float(reward[0]))
			plot_rewards()
			break

print(np.mean(episode_rewards[-3000:]))
# should ideally be around -0.0363752
# was -0.14883333333333335 after 20k episodes
print('Complete')
plt.ioff()
plt.show()




