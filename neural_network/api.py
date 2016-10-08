from neural_network import *
import pickle 

class Api:
	network_file = "network"
	network = None

	def load_network(self):
		with open(self.network_file, 'rb') as pickle_file:
			network = pickle.load(pickle_file)
		return network

	def save_network(self, network):
		with open(self.network_file, 'wb') as pickle_file:
			pickle.dump(network, pickle_file)

	def train(self, labeledExamples):	
		network = self.load_network()
		network.train(labeledExamples, maxIterations=5000)
		self.save_network(network)

	def evaluate(self, example):
		network = self.load_network()
		return network.evaluate(example)

	def create(self):
		self.network = makeNetwork(3, 1, 3)
		self.save_network(self.network)