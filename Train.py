
import re
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TransE import TransE
from readTrainingData import readData
from generatePosAndCorBatch import generateBatches, dataset


class trainTransE:

    def __init__(self):
        self.inAdd = "./data/FB15K"
        self.outAdd = "./data/outputData"
        self.preAdd = "./data/outputData"
        self.preOrNot = False
        self.entityDimension = 100
        self.relationDimension = 100
        self.numOfEpochs = 1000
        self.outputFreq = 50
        self.numOfBatches = 100
        self.learningRate = 0.01  # 0.01
        self.weight_decay = 0.001  # 0.005  0.02
        self.margin = 1.0
        self.norm = 2
        self.top = 10
        self.patience = 10
        self.earlyStopPatience = 5
        self.bestAvFiMR = None

        self.train2id = {}
        self.trainTriple = None
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.nums = [0, 0, 0]
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        self.headRelation2Tail = {}
        self.tailRelation2Head = {}
        self.positiveBatch = {}
        self.corruptedBatch = {}
        self.entityEmbedding = None
        self.relationEmbedding = None

        self.validate2id = {}
        self.validateHead = None
        self.validateRelation = None
        self.validateTail = None
        self.numOfValidateTriple = 0

        self.test2id = {}
        self.testHead = None
        self.testRelation = None
        self.testTail = None
        self.numOfTestTriple = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.start()
        self.train()
        self.end()


    def start(self):
        print("-----Training Started at {}-----".format(time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(time.time()))))
        print("Input address: {}".format(self.inAdd))
        print("Output address: {}".format(self.outAdd))
        print("Entity dimension: {}".format(self.entityDimension))
        print("Relation dimension: {}".format(self.relationDimension))
        print("#(epochs): {}".format(self.numOfEpochs))
        print("output training results every {} epochs".format(self.outputFreq))
        print("#(batches): {}".format(self.numOfBatches))
        print("learning rate: {}".format(self.learningRate))
        print("weight decay: {}".format(self.weight_decay))
        print("margin: {}".format(self.margin))
        print("norm: {}".format(self.norm))
        print("is a continued learning: {}".format(self.preOrNot))
        if self.preOrNot:
            print("pre-trained result address: {}".format(self.preAdd))
        print("device: {}".format(self.device))
        print("patience: {}".format(self.patience))
        print("early stop patience: {}".format(self.earlyStopPatience))

    def end(self):
        print("-----Training Finished at {}-----".format(time.strftime('%m-%d-%Y %H:%M:%S',time.localtime(time.time()))))

    def train(self):
        read = readData(self.inAdd, self.train2id, self.headRelation2Tail, self.tailRelation2Head,
                      self.entity2id, self.id2entity, self.relation2id, self.id2relation, self.nums)
        self.trainTriple = read.out()
        self.numOfTriple = self.nums[0]
        self.numOfEntity = self.nums[1]
        self.numOfRelation = self.nums[2]

        self.readValidateTriples()
        self.readTestTriples()

        transE = TransE(self.numOfEntity, self.numOfRelation, self.entityDimension, self.relationDimension, self.margin,
                        self.norm)

        if self.preOrNot:
            self.preRead(transE)

        transE.to(self.device)

        self.bestAvFiMR = self.validate(transE)
        self.entityEmbedding = transE.entity_embeddings.weight.data.clone()
        self.relationEmbedding = transE.relation_embeddings.weight.data.clone()

        criterion = nn.MarginRankingLoss(self.margin, False).to(self.device)
        optimizer = optim.SGD(transE.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)

        dataSet = dataset(self.numOfTriple)
        batchSize = float(self.numOfTriple / self.numOfBatches)
        dataLoader = DataLoader(dataSet, batchSize, True)

        patienceCount = 0

        for epoch in range(self.numOfEpochs):
            epochLoss = 0
            for batch in dataLoader:
                self.positiveBatch = {}
                self.corruptedBatch = {}
                generateBatches(batch, self.train2id, self.positiveBatch, self.corruptedBatch, self.numOfEntity,
                                self.headRelation2Tail, self.tailRelation2Head)
                optimizer.zero_grad()
                positiveBatchHead = self.positiveBatch["h"].to(self.device)
                positiveBatchRelation = self.positiveBatch["r"].to(self.device)
                positiveBatchTail = self.positiveBatch["t"].to(self.device)
                corruptedBatchHead = self.corruptedBatch["h"].to(self.device)
                corruptedBatchRelation = self.corruptedBatch["r"].to(self.device)
                corruptedBatchTail = self.corruptedBatch["t"].to(self.device)
                output = transE(positiveBatchHead, positiveBatchRelation, positiveBatchTail, corruptedBatchHead,
                                   corruptedBatchRelation, corruptedBatchTail)
                positiveLoss, negativeLoss = output.view(2, -1)
                tmpTensor = torch.tensor([-1], dtype=torch.float).to(self.device)
                batchLoss = criterion(positiveLoss, negativeLoss, tmpTensor)
                batchLoss.backward()
                optimizer.step()
                epochLoss += batchLoss

            print("epoch: {}, loss: {}".format(epoch, epochLoss))

            tmpAvFiMR = self.validate(transE)

            if tmpAvFiMR < self.bestAvFiMR:
                print("best averaged raw mean rank: {} -> {}".format(self.bestAvFiMR, tmpAvFiMR))
                patienceCount = 0
                self.bestAvFiMR = tmpAvFiMR
                self.entityEmbedding = transE.entity_embeddings.weight.data.clone()
                self.relationEmbedding = transE.relation_embeddings.weight.data.clone()
            else:
                patienceCount += 1
                print("early stop patience: {}, patience count: {}, current rank: {}, best rank: {}".format(self.earlyStopPatience, patienceCount, tmpAvFiMR, self.bestAvFiMR))
                if patienceCount == self.patience:
                    if self.earlyStopPatience == 1:
                        break
                    print("learning rate: {} -> {}".format(self.learningRate, self.learningRate/2))
                    print("weight decay: {} -> {}".format(self.weight_decay, self.weight_decay*2))
                    self.learningRate /= 2
                    self.weight_decay *= 2
                    transE.entity_embeddings.weight.data = self.entityEmbedding.clone()
                    transE.relation_embeddings.weight.data = self.relationEmbedding.clone()
                    optimizer = optim.SGD(transE.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)
                    patienceCount = 0
                    self.earlyStopPatience -= 1

            if (epoch+1)%self.outputFreq == 0 or (epoch+1) == self.numOfEpochs:
                self.write()
            print()

        transE.entity_embeddings.weight.data = self.entityEmbedding.clone()
        transE.relation_embeddings.weight.data = self.relationEmbedding.clone()
        self.test(transE)
        # self.fastTest(transE)

    def validate(self, transE):
        meanRank = 0
        for tmpTriple in range(self.numOfValidateTriple):
            meanRank += transE.fastValidate(self.validateHead[tmpTriple], self.validateRelation[tmpTriple], self.validateTail[tmpTriple])
        return meanRank/self.numOfValidateTriple

    def fastTest(self, transE):  # Massive memory is required
        print("-----Fast Test Started at {}-----".format(time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))
        meanRank = torch.tensor([0., 0.]).to(self.device)
        transE.fastTest(meanRank, self.testHead, self.testRelation, self.testTail,
                    self.trainTriple.to(self.device), self.numOfTestTriple)
        print("-----Result of Link Prediction (Raw)-----")
        print("|  Mean Rank  |  Filter@{}  |".format(self.top))
        print("|  {}  |  under implementing  |".format(meanRank[0]))
        print("-----Result of Link Prediction (Filter)-----")
        print("|  Mean Rank  |  Filter@{}  |".format(self.top))
        print("|  {}  |  under implementing  |".format(meanRank[1]))
        print("-----Fast Test Ended at {}-----".format(time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))

    def test(self, transE):
        print("-----Test Started at {}-----".format(time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))
        meanRank = torch.tensor([0., 0.]).to(self.device)
        rMR = 0.
        fMR = 0.
        rHit = 0.
        fHit = 0.
        for tmpTriple in range(self.numOfTestTriple):
            if (tmpTriple+1)%1000 == 0:
                print("{} test triples processed!".format(tmpTriple+1))
            transE.test(meanRank, self.testHead[tmpTriple], self.testRelation[tmpTriple], self.testTail[tmpTriple], self.trainTriple.to(self.device))
            rMR += meanRank[0]
            fMR += meanRank[1]
            if meanRank[0] <= self.top:
                rHit += 1
            if meanRank[1] <= self.top:
                fHit += 1
        print("-----Result of Link Prediction (Raw)-----")
        print("|  Mean Rank  |  Filter@{}  |".format(self.top))
        print("|  {}  |  {}  |".format(rMR/self.numOfTestTriple, rHit/self.numOfTestTriple))
        print("-----Result of Link Prediction (Filter)-----")
        print("|  Mean Rank  |  Filter@{}  |".format(self.top))
        print("|  {}  |  {}  |".format(fMR/self.numOfTestTriple, fHit/self.numOfTestTriple))
        print("-----Test Ended at {}-----".format(time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time()))))

    def write(self):
        print("-----Writing Training Results to {}-----".format(self.outAdd))

        # Entities
        entity2vecAdd = self.outAdd + "/entity2vec.pickle"
        entityOutput = open(entity2vecAdd, "w")
        pickle.dump(self.entityEmbedding, entityOutput)
        entityOutput.close()

        # Relations
        relation2vecAdd = self.outAdd + "/relation2vec.pickle"
        relationOutput = open(relation2vecAdd, "w")
        pickle.dump(self.relationEmbedding, relationOutput)
        relationOutput.close()

    def preRead(self, transE):
        print("-----Reading Pre-Trained Results from {}-----".format(self.preAdd))

        # Entities
        entityInput = open(os.path.join(self.preAdd, "entity2vec.pickle"), "r")
        tmpEntityEmbedding = pickle.load(entityInput)
        entityInput.close()
        transE.entity_embeddings.weight.data = tmpEntityEmbedding

        # Relations
        relationInput = open(os.path.join(self.preAdd, "relation2vec.pickle"), "r")
        tmpRelationEmbedding = pickle.load(relationInput)
        relationInput.close()
        transE.relation_embeddings.weight.data = tmpRelationEmbedding

    def readValidateTriples(self):
        print("-----Reading valid2id.txt from {} -----".format(self.inAdd))
        count = 0
        self.validate2id["h"] = []
        self.validate2id["r"] = []
        self.validate2id["t"] = []
        inputData = open(os.path.join(self.inAdd, "valid2id.txt"))
        line = inputData.readline()
        self.numOfValidateTriple = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                tmpTail = int(re.findall(r"\d+", line)[1])
                tmpRelation = int(re.findall(r"\d+", line)[2])
                self.validate2id["h"].append(tmpHead)
                self.validate2id["r"].append(tmpRelation)
                self.validate2id["t"].append(tmpTail)
                count += 1
            else:
                print("error in valid2id.txt at Line {}".format(count + 2))
            line = inputData.readline()
        inputData.close()
        if count == self.numOfValidateTriple:
            print("number of validation triples: {}".format(self.numOfValidateTriple))
            self.validateHead = torch.LongTensor(self.validate2id["h"]).to(self.device)
            self.validateRelation = torch.LongTensor(self.validate2id["r"]).to(self.device)
            self.validateTail = torch.LongTensor(self.validate2id["t"]).to(self.device)
        else:
            print("count: {}".format(count))
            print("expected number of validation triples: {}".format(self.numOfValidateTriple))
            print("error in valid2id.txt")

    def readTestTriples(self):
        print("-----Reading test2id.txt from {}-----".format(self.inAdd))
        count = 0
        self.test2id["h"] = []
        self.test2id["r"] = []
        self.test2id["t"] = []
        inputData = open(os.path.join(self.inAdd, "test2id.txt"))
        line = inputData.readline()
        self.numOfTestTriple = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                tmpTail = int(re.findall(r"\d+", line)[1])
                tmpRelation = int(re.findall(r"\d+", line)[2])
                self.test2id["h"].append(tmpHead)
                self.test2id["r"].append(tmpRelation)
                self.test2id["t"].append(tmpTail)
                count += 1
            else:
                print("error in test2id.txt at Line {}".format(count + 2))
            line = inputData.readline()
        inputData.close()
        if count == self.numOfTestTriple:
            print("number of test triples: {}".format(self.numOfTestTriple))
            self.testHead = torch.LongTensor(self.test2id["h"]).to(self.device)
            self.testRelation = torch.LongTensor(self.test2id["r"]).to(self.device)
            self.testTail = torch.LongTensor(self.test2id["t"]).to(self.device)
        else:
            print("count: {}".format(count))
            print("expected number of test triples: {}".format(self.numOfTestTriple))
            print("error in test2id.txt")

if __name__ == '__main__':
    trainTransE = trainTransE()

