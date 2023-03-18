import pickle
import functions
import models
import random
import torch
import os
import torch.optim as optim
import transformerBase
import tokenizer
from torch.utils.data import DataLoader
import datasetClasses

## global variables for project
### change here to run on cluster ####
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"

# load tokenizer and model
Tokenizer = tokenizer.tokenizer()
## args: hiddenLenc, attLayers, attentionHeads, device, Training=True, predictionInterval=None)
model = transformerBase.Transformer(1000, 1, 1, device, predictionInterval=4).to(device)

# load weights to tokenizer
os.chdir(pathOrigin + "/models")
Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
Tokenizer = Tokenizer.to(device)


# define hyperparameters
params = {"learningRate": 0.001, "weightDecay": 0.01, "epochs": 100, "batchSize": 50, "optimizer": "adam", "validationStep": 1}

# get dataLoaders
datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "trainData"))
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "valData"))
dataVal = DataLoader(datasetVal, 1, shuffle = True)

# criterion
loss = torch.nn.MSELoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoop(dataTrain, dataVal, Tokenizer,  model, loss, False, "Transformer", params, True, device)




