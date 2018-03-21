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

if state_machine.terminal:
	episode_rewards.append(float(reward[0]))
	plot_rewards()
	break