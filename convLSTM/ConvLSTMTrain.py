import ConvLSTM
import functions
import os
from torch.utils.data import DataLoader
import datasetClasses
import torch

# global
device = "cuda"

model = ConvLSTM.ConvLSTMFuturePredictor(1, 20, (20, 50, 50), 3).to(device)

# define hyperparameters
params = {"learningRate": 0.1, "weightDecay": 0.01, "epochs": 20, "batchSize": 20, "optimizer": "adam", "validationStep": 10}

# get dataLoaders
datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)

# criterion
loss = torch.nn.MSELoss()

## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoopConvLSTM(dataTrain, dataVal, None,  model, loss, False, "ConvLSTM", params, True, device)