import numpy as np

class BlackjackSM:
	def __init__(self):
		# dealer hits on soft 17
		self.rule = list(range(4,17)) + list(range(31,38))
		# designated integers for soft hands
		self.soft = list(range(32,41))

		# 2-20 are hard hands, 31-40 are soft hands
		self.self_space = list(range(4,21)) + self.soft
		# whatever value the deal has showing
		self.dealer_space = list(range(0,11))

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
		self.dealer_hand = draws[2] 

		# no point in considering player blackjack
		if self.hand == 21:
			self.hand = self.hand - np.random.randint(1,10)

		self.bet = 1.
		self.terminal = False
		# tracks whether this is the first action for doubling down
		self.new = True

	def get_actions(self):
		if self.terminal:
			return []
		output = list(range(2))
		if self.new:
			output += [2]
			if self.pair:
				output += [3]
		return output

	def hit(self):
		new_card = self.draw()
		self.hand = self.calculate_hand(self.hand, new_card)
		if self.hand == 21:
			self.terminal = True
		if self.hand > 21 and self.hand < 31:
			self.bet = -self.bet
			self.terminal = True
		self.new = False

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
		self.hand /= 2
		self.bet *= 2
		new_card = self.draw()
		if new_card != self.hand:
			self.pair = False
		self.hand = self.calculate_hand(self.hand, new_card)

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
		self.dealer_hand = self.calculate_hand(self.dealer_hand, new_card)
		while self.dealer_hand in self.rule:
			new_card = self.draw()
			self.dealer_hand = self.calculate_hand(self.dealer_hand, new_card)

	def get_reward(self):
		if not self.terminal:
			return 0
		if self.bet <= 0:
			return self.bet

		self.complete()
		if self.dealer_hand == 41:
			return -self.bet
		if self.dealer_hand > 21 and self.dealer_hand < 31:
			return self.bet

		if self.hand in self.soft:
			self.hand -= 20
		if self.dealer_hand in self.soft:
			self.dealer_hand -= 20

		if self.hand == self.dealer_hand:
			return 0
		if self.hand < self.dealer_hand:
			return -self.bet
		if self.hand > self.dealer_hand:
			return self.bet

	def state(self):
		return (self.hand, self.dealer_hand)

# state = BlackjackSM()
# state.new_hand()
# print(str(state.hand) + " vs. " + str(state.dealer_hand))
# actions = state.get_actions()
# print(actions)
# if 3 in actions:
# 	state.split()
# else:
# 	state.double()
# print(state.get_actions())
# state.stand()
# reward = state.get_reward()
# print(str(state.hand) + " vs. " + str(state.dealer_hand))
# print(reward)
# state.new_hand()
# print(str(state.hand) + " vs. " + str(state.dealer_hand))
# actions = state.get_actions()
# print(actions)

