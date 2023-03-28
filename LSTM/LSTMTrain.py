import LSTM
import functions
import torch
import os
import transformerBase
import tokenizer
from torch.utils.data import DataLoader
import datasetClasses

## global variables for project
### change here to run on cluster ####
pathOrigin = "/mnt/qb/work/ludwig/lqb875"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"

# load tokenizer and model
#Tokenizer = tokenizer.tokenizerGodspeed()

# args: lstmLayers, lstmHiddenSize, lstmInputSize, dropout
model = LSTM.LSTM(10,10, 2500, 2500, 0.3, device).to(device)

# load weights to tokenizer
#os.chdir(pathOrigin + "/models")
#Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizerConv")
#Tokenizer = Tokenizer.to(device)


# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.01, "epochs": 2, "batchSize": 50, "optimizer": "adam", "validationStep": 1}


# get dataLoaders
#datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)


# criterion
loss = torch.nn.MSELoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
#functions.trainLoopLSTM(dataTrain, dataVal, Tokenizer,  model, loss, False, "LSTMConvTok", params, True, device)
functions.trainLoop(dataTrain, dataVal, None,  model, loss, False, "LSTMEncDec", params, True, device)




