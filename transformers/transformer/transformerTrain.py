import pickle
import functions
import random
import torch
import os
import torch.optim as optim
import transformerBase
import tokenizer
from torch.utils.data import DataLoader
import datasetClasses
import LSTM

## global variables for project
### change here to run on cluster ####
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"

# load tokenizer and model
Tokenizer = tokenizer.tokenizerExp().to(device)
## args: hiddenLenc, attLayers, attentionHeads, device, Training=True, predictionInterval=None)
model = transformerBase.Transformer(1600, 1, 1, device, predictionInterval=4).to(device)


# load weights to tokenizer
#os.chdir(pathOrigin + "/models")
#Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
#Tokenizer = Tokenizer.to(device)


# define hyperparameters
params = {"learningRate": 0.001, "weightDecay": 0.01, "epochs": 5, "batchSize": 100, "optimizer": "adam", "validationStep": 1}

# get dataLoaders /home/jonas/datasets/parbati
#datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
dataVal = DataLoader(datasetVal, 10, shuffle = True)

# criterion
loss = torch.nn.MSELoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoop(dataTrain, dataVal, Tokenizer,  model, loss, False, "TransformerMiddle", params, True, device)
#functions.trainLoop(dataTrain, dataVal, None,  model, loss, False, "TransformerMiddle", params, True, device)




