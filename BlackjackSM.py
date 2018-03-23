import numpy as np

class BlackjackSM:
	def __init__(self):
		# dealer hits on soft 17
		self.rule = list(range(4,17)) + list(range(32,38))
		# designated range for soft hands
		self.soft = list(range(32,41))
		self.len_state = 5
		# 0 is hit, 1 is stand, 2 is surrender, 3 is double, 4 is split
		self.len_actions = 5

	def draw(self):
		card = np.random.randint(1,13)
		if card > 10:
			return 10
		return card

	def calculate_hand(self, hand, new_card):
		if hand == 1: 
			if new_card == 10:
				return 21
			return new_card + 31
		if new_card == 1:
			if hand < 10: 
				return hand + 31
			if hand == 40 or hand == 10:
				return 21
			return hand + 1
		if hand < 21:
			return hand + new_card
		if (hand + new_card) > 41:
			return hand + new_card - 30
		elif (hand + new_card) == 41:
			return 21
		return hand + new_card

	def new_hand(self):
		draws = [self.draw() for _ in range(3)]
		self.pair = (draws[0] == draws[1])
		self.hand = self.calculate_hand(draws[0],draws[1])
		if draws[2] == 1:
			self.dealer_hand = 11
		else:
			self.dealer_hand = draws[2] 

		# no point in considering player blackjack
		if self.hand == 21:
			self.hand = self.hand - np.random.randint(1,10)

		self.bet = 1.
		self.terminal = False
		# tracks whether this is the first action for doubling down
		self.new = True
		# tracks whether we've completed the dealer's hand to get a reward
		self.completed = False

	def actions(self):
		if self.terminal:
			return []
		output = list(range(2))
		if self.new:
			output += [2,3]
			if self.pair:
				output += [4]
		return output

	def mask(self): 
		mask = np.zeros(self.len_actions, dtype=int)
		mask[self.actions()] = 1
		return mask

	def mask_for(self, state):
		mask = np.ones(self.len_actions, dtype=int)
		if state[2] != 1:
			mask[3] = 0
		if state[3] != 1:
			mask[4] = 0
		return mask

	def hit(self):
		new_card = self.draw()
		self.hand = self.calculate_hand(self.hand, new_card)
		if self.hand == 21:
			self.terminal = True
		if self.hand > 21 and self.hand < 31:
			self.bet = -self.bet
			self.terminal = True
		self.new = False
		self.pair = False

	def stand(self):
		self.terminal = True

	def surrender(self):
		self.bet = -self.bet / 2
		self.stand()

	def double(self):
		self.bet *= 2
		self.hit()
		self.stand()

	def split(self):
		if self.hand == 32:
			self.hand = 1
		else:
			self.hand //= 2
		self.bet *= 2
		new_card = self.draw()
		if new_card != self.hand:
			self.pair = False
		self.hand = self.calculate_hand(self.hand, new_card)
		if self.hand == 21:
			self.terminal = True

	def do(self, action):
		if action == 0:
			self.hit()
		elif action == 1:
			self.stand()
		elif action == 2:
			self.surrender()
		elif action == 3:
			self.double()
		elif action == 4:
			self.split()
		else:
			return False

	def complete(self):
		new_card = self.draw()
		if (self.dealer_hand == 1 and new_card == 10) or (self.dealer_hand == 10 and new_card == 1):
			self.dealer_hand = 41
			return
		if self.dealer_hand == 11:
			self.dealer_hand = 1
		self.dealer_hand = self.calculate_hand(self.dealer_hand, new_card)
		while self.dealer_hand in self.rule:
			new_card = self.draw()
			self.dealer_hand = self.calculate_hand(self.dealer_hand, new_card)
		self.completed = True

	def reward(self):
		if not self.terminal:
			return 0
		if self.bet <= 0:
			return self.bet

		if not self.completed:
			self.complete()

		if self.dealer_hand == 41:
			return -1

		if self.hand in self.soft:
			self.hand -= 20
		if self.dealer_hand in self.soft:
			self.dealer_hand -= 20
		
		if self.dealer_hand > 21:
			return self.bet
		if self.hand == self.dealer_hand:
			return 0
		if self.hand < self.dealer_hand:
			return -self.bet
		if self.hand > self.dealer_hand:
			return self.bet

	def state(self):
		if self.hand in self.soft: 
			return tuple(map(int,[self.hand - 20, 1, 1*self.new, 1*self.pair, self.dealer_hand]))
		else:
			return tuple(map(int,[self.hand, 0, 1*self.new, 1*self.pair, self.dealer_hand]))

	def set_state(self,state):
		self.hand = state[0] + 20*state[1]
		self.dealer_hand = state[4]
		self.new = state[2]
		self.pair = state[3]

		self.bet = 1
		self.terminal = False
		self.completed = False

# sm = BlackjackSM()
# num_episodes = 10000
# avg_reward_double = 0
# avg_reward_hit = 0
# state = (11,0,1,0,4)
# for _ in range(num_episodes):
# 	sm.set_state(state)
# 	sm.surrender()
# 	avg_reward_double += sm.reward() / num_episodes
# print(avg_reward_double)
# for _ in range(num_episodes):
# 	sm.set_state(state)
# 	sm.hit()
# 	sm.stand()
# 	avg_reward_hit += sm.reward() / num_episodes
# print(avg_reward_hit)
