import numpy as np
from PNode import *


class CNode:
    def __init__(self, numWeights):
        self.dWeights = np.random.normal(0, 1, numWeights)
        self.PNodes = []

    def CalculateDistance(self, InputNode):
        distance = 0
        for i in range(len(InputNode)):
            distance += (InputNode[i] - self.dWeights[i]) * (InputNode[i] - self.dWeights[i])
        return distance

    def AdjustWeights(self, target, LearningRate, Influence):
        for i in range(len(target)):
            self.dWeights[i] += LearningRate * Influence * (target[i] - self.dWeights[i])

    def addPNode(self, corpus, inputPNode):
        self.PNodes.append(PNode(corpus, inputPNode))
