import LSTM
import functions
import torch
import os
from torch.utils.data import DataLoader
import datasetClasses

## global variables for project
### change here to run on cluster ####
pathOrigin = "/mnt/qb/work/ludwig/lqb875"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"


# args: lstmLayers, lstmHiddenSize, lstmInputSize, dropout
model = LSTM.LSTM(3,3, 2500, 2500, 0.1, device).to(device)


# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 40, "batchSize": 100, "optimizer": "adam", "validationStep": 100}


# get dataLoaders
#datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "val")
#datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)


# criterion
loss = torch.nn.MSELoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoop(dataTrain, dataVal,  model, loss, False, "LSTMEncDec", params, True, device)




