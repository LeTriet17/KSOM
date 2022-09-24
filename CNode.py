import numpy as np
from PNode import *
import math

class CNode:
    def __init__(self, numWeights):
        self.dWeights = np.random.normal(0, 1, numWeights)
        self.PNodes = {}
        self.represent_vector=None
        # self.idx=None

    def CalculateDistance(self, InputNode):
        temp = self.dWeights - InputNode
        sum_sq = np.dot(temp.T, temp)
        return sum_sq

    def CalculateDistance2CNode(self, InputCNode):
        temp=self.dWeights-InputCNode.dWeights
        sum_sq = np.dot(temp.T, temp)
        return math.sqrt(sum_sq)


    def AdjustWeights(self, target, LearningRate, Influence):
            self.dWeights += LearningRate * Influence * (target - self.dWeights)

    def addPNode(self, corpus, inputPNode, idx):
        self.PNodes[idx]=(PNode(corpus, inputPNode))
        return
    
    def __str__(self):
      return np.array2string(self.dWeights)
