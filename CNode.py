import numpy as np
class CNode:
    def __init__(self, numWeights):
        self.dWeights = np.random.normal(0, 1, numWeights)
        self.PNode =[]
        
    def CalculateDistance(self,InputNode):
        distance = 0
        for i in range(len(InputNode)):
            distance+=(InputNode[i] - self.dWeights[i]) * (InputNode[i] - self.dWeights[i])
        return distance
    
    def AdjustWeights(self,target,LearningRate,Influence):
        for i in range(len(target)):
            self.dWeights[i] += LearningRate * Influence * (target[i] - self.dWeights[i])

    def addPNode(self, PNode):
        self.PNode.append(PNode)
