import random
import math
import numpy as np


def activationFunction(input):
	return 1.0/(1.0+np.exp(-input))


class Node:

	def __init__(self):
		self.lastOutput = None
		self.lastInput = None
		self.error = None
		self.outgoingEdges = []
		self.incomingEdges = []
		self.addBias()

	def addBias(self):
		self.incomingEdges.append(Edge(BiasNode(), self))

	def evaluate(self, inputVector):
		self.lastInput = []
		weightedSum = 0

		for e in self.incomingEdges:
			theInput = e.source.evaluate(inputVector)
			self.lastInput.append(theInput)
			weightedSum += e.weight * theInput
		
		self.lastOutput = activationFunction(weightedSum)
		return self.lastOutput

	def getError(self, label):
		if self.outgoingEdges == []: # this is an output node
			self.error = label - self.lastOutput
			print(self.error)
		else: 
			self.error = sum([edge.weight * edge.target.getError(label) for edge in self.outgoingEdges])
		return self.error

	def updateWeights(self, learningRate):
		if (self.error is not None 
			and self.lastOutput is not None
			and self.lastInput is not None):
			
			for i, edge in enumerate(self.incomingEdges):
				edge.weight += (learningRate * self.lastOutput * (1 - self.lastOutput) * self.error * self.lastInput[i])

			for edge in self.outgoingEdges:
				edge.target.updateWeights(learningRate)

			self.error = None
			self.lastInput = None
			self.lastOutput = None

class InputNode(Node):
	def __init__(self, index):
		Node.__init__(self)
		self.index = index # the index of the input vector corresponding to this node

	def evaluate(self, inputVector):
		self.lastOutput = inputVector[self.index]
		return self.lastOutput

	def updateWeights(self, learningRate):
		for edge in self.outgoingEdges:
			edge.target.updateWeights(learningRate)

	def getError(self, label):
		for edge in self.outgoingEdges:
			edge.target.getError(label)


class BiasNode(InputNode):
	def __init__(self):
		Node.__init__(self)

	def evaluate(self, inputVector):
		return 1.0

	def addBias(self):
		pass


class Edge:
	def __init__(self, source, target):
		self.weight = random.uniform(0,1)
		self.source = source
		self.target = target

		# attach the edges to its nodes
		source.outgoingEdges.append(self)
		target.incomingEdges.append(self)


class Network:
	def __init__(self):
		self.inputNodes = []
		self.outputNode = None

	def evaluate(self, inputVector):
		return self.outputNode.evaluate(inputVector)

	def propagateError(self, label):
		for node in self.inputNodes:
			node.getError(label)

	def updateWeights(self, learningRate):
		for node in self.inputNodes:
			node.updateWeights(learningRate)

	def train(self, labeledExamples, learningRate=0.9, maxIterations=10000):
		while maxIterations > 0:
			for example, label in labeledExamples:
				output = self.evaluate(example)
				self.propagateError(label)
				self.updateWeights(learningRate)
				maxIterations -= 1


def makeNetwork(numInputs, numHiddenLayers, numInEachLayer):
	network = Network()
	inputNodes = [InputNode(i) for i in range(numInputs)]
	outputNode = Node()
	network.outputNode = outputNode
	network.inputNodes.extend(inputNodes)

	layers = [[Node() for _ in range(numInEachLayer)] for _ in range(numHiddenLayers)]

	# connect inputs to first hidden layer
	for inputNode in inputNodes:
		for node in layers[0]:
			Edge(inputNode, node)

	# connect hidden layers pair-wise
	for layer1, layer2 in [(layers[i], layers[i+1]) for i in range(numHiddenLayers-1)]:
		for node1 in layer1:
			for node2 in layer2:
				Edge(node1, node2)

	# connect last hidden layer to output node
	for node in layers[-1]:
		Edge(node, outputNode)

	return network
