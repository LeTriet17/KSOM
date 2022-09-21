import CNode
import math
import numpy as np
import copy
from CNode import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

def identity_tokenizer(text):
    return text


class CSom:
    def __init__(self, MapSize, corpus, numIterations, constStartLearningRate=0.5):
        self.MapSize = MapSize
        self.corpus = corpus
        self.numIterations = numIterations
        self.dMapRadius = MapSize / 2
        self.dTimeConstant = numIterations / math.log(self.dMapRadius)
        self.dLearningRate = constStartLearningRate
        self.constLearningRate = constStartLearningRate
        print("Start TFIDF")
        PNodes = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
        self.PNodes = PNodes.fit_transform(corpus).todense()
        print("Done TFIDF")
        print(self.PNodes[1])
        NodeDimension = self.PNodes.shape[1]
        self.m_Som = np.asarray([[CNode(NodeDimension) for j in range(MapSize)] for i in range(MapSize)])

    def FindBestMatchingNode(self, inputPNode):
        LowestDistance = 999999
        SecDistance = 999999
        winner = None
        PNode = np.squeeze(np.asarray(inputPNode))
        for iy, ix in np.ndindex(self.m_Som.shape):
            dist = self.m_Som[iy, ix].CalculateDistance(PNode)
            if dist < LowestDistance:
                # if len(self.m_Som[iy, ix].PNodes) > 0:
                #     totalSim = 0
                #     for i in range(len(self.m_Som[iy, ix].PNodes)):
                #         totalSim += int(cosine_similarity(inputPNode, self.m_Som[iy, ix].PNodes[i].vector))
                #     if totalSim / len(self.m_Som[iy, ix].PNodes) < 0.5:
                #         continue
                LowestDistance = dist
                winner = self.m_Som[iy, ix]
                win_x=ix
                win_y=iy
        return winner, ix, iy

    def Train(self):
        print("Start Training")
        for i in range(self.numIterations):
            print(f"Epoch {i}")
            randomPNode = int(np.random.randint(self.PNodes.shape[0], size=1))
            WinningNode, grid_x, grid_y = self.FindBestMatchingNode(self.PNodes[randomPNode])
            dNeighbourhoodRadius = self.dMapRadius * math.exp(-float(i) / self.dTimeConstant);
            WidthSq = dNeighbourhoodRadius * dNeighbourhoodRadius
            for iy, ix in np.ndindex(self.m_Som.shape):
                DistToNodeSq = (iy-grid_y)*(iy-grid_y)+(ix-grid_x)*(ix-grid_x)

                if True:
                    self.dInfluence = math.exp(-(DistToNodeSq) / (2 * WidthSq))
                    self.m_Som[iy, ix].AdjustWeights(np.squeeze(np.asarray(self.PNodes[randomPNode])),
                                                     self.dLearningRate, self.dInfluence)
            self.dLearningRate = self.constLearningRate * math.exp(-float(i) / (self.dTimeConstant))
            if i%5==0:
              with open('ksom.txt', 'w') as f:
                for iy, ix in np.ndindex(self.m_Som.shape):
                  f.write(f"{iy} {ix} {self.m_Som[iy,ix]} \n")


        # for i in range(self.PNodes.shape[0]):
        #     SuitNode = self.FindBestMatchingNode(self.PNodes[i])
        #     # print(self.PNodes[i])
        #     # print(SuitNode)
        #     SuitNode.addPNode(self.corpus[i], self.PNodes[i])

    def Plot(self):
        plt.rcParams["figure.autolayout"] = True
        count = 0
        data2D = np.arange(self.MapSize ** 2).reshape(self.MapSize, self.MapSize)
        for iy, ix in np.ndindex(self.m_Som.shape):
            if len(self.m_Som[iy, ix].PNodes) == 0:
                data2D[iy, ix] = 0
        cmap = plt.cm.get_cmap('Blues').copy()
        cmap.set_under('white')
        plt.imshow(data2D, cmap=cmap, vmin=0)
        plt.colorbar(extend='min', extendrect=True)
        plt.show()

