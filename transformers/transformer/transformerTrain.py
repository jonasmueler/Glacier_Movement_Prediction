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
import transformerLSTM

## global variables for project
### change here to run on cluster ####
pathOrigin = "/mnt/qb/work/ludwig/lqb875"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"

# load tokenizer and model
#Tokenizer = tokenizer.tokenizer()
## args: hiddenLenc, attLayers, attentionHeads, device, Training=True, predictionInterval=None)
model = transformerBase.Transformer(2500, 5, 10, device, predictionInterval=4).to(device)
#model = transformerLSTM.Transformer(2500, 2, 4, device, predictionInterval=4).to(device)

# load weights to tokenizer
#os.chdir(pathOrigin + "/models")
#Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizerConv")
#Tokenizer = Tokenizer.to(device)


# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 30, "batchSize": 100, "optimizer": "adam", "validationStep": 100}

# get dataLoaders
datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)

# criterion
loss = torch.nn.MSELoss()
#loss = torch.nn.CrossEntropyLoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoopUnet(dataTrain, dataVal,  model, loss, False, "TransformerLanguage", params, True, device)
                     #  (trainLoader, valLoader, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin)
#functions.trainLoopClassification(dataTrain, dataVal, None,  model, loss, False, "TransformerBigClassification", params, True, device)



