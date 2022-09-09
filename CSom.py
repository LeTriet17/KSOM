import CNode
import math
import numpy as np
import copy
from CNode import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class CSom:
    def __init__(self, MapSize, corpus, numIterations, constStartLearningRate=0.1):
        self.corpus = copy.copy(corpus)
        self.numIterations = numIterations
        self.dMapRadius = MapSize / 2
        self.dTimeConstant = numIterations / math.log(self.dMapRadius)
        self.dLearningRate = constStartLearningRate
        PNodes = TfidfVectorizer()
        self.PNodes = PNodes.fit_transform(corpus).todense()
        NodeDimension = self.PNodes.shape[1]
        self.m_Som = np.asarray([[CNode(NodeDimension) for j in range(MapSize)] for i in range(MapSize)])

    def FindBestMatchingNode(self, inputPNode):
        LowestDistance = 999999
        winner = None
        PNode = np.squeeze(np.asarray(inputPNode))
        for iy, ix in np.ndindex(self.m_Som.shape):
            dist = self.m_Som[iy, ix].CalculateDistance(PNode)
            if dist < LowestDistance:
                LowestDistance = dist
                winner = self.m_Som[iy, ix]
        return winner

    def Train(self):
        for i in range(self.numIterations):
            randomPNode = int(np.random.randint(self.PNodes.shape[0], size=1))
            WinningNode = self.FindBestMatchingNode(self.PNodes[randomPNode])
            dNeighbourhoodRadius = self.dMapRadius * math.exp(-float(i) / self.dTimeConstant);
            WidthSq = dNeighbourhoodRadius * dNeighbourhoodRadius
            for iy, ix in np.ndindex(self.m_Som.shape):
                DistToNodeSq = self.m_Som[iy, ix].CalculateDistance(WinningNode.dWeights)

                if DistToNodeSq < dNeighbourhoodRadius ** 2:
                    self.dInfluence = math.exp(-(DistToNodeSq) / (2 * WidthSq))
                    self.m_Som[iy, ix].AdjustWeights(np.squeeze(np.asarray(self.PNodes[randomPNode])),
                                                     self.dLearningRate, self.dInfluence)
            self.dLearningRate = self.dLearningRate * math.exp(-float(i) / (self.numIterations - i))
        for i in range(self.PNodes.shape[0]):
            SuitNode = self.FindBestMatchingNode(self.PNodes[i])
            SuitNode.addPNode(self.corpus[i],self.PNodes[i])
