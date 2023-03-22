import LSTM
import functions
import torch
import os
import transformerBase
import tokenizer
from torch.utils.data import DataLoader
import datasetClasses


#### this is ugly, fix it!!!
# fix dataloader


## global variables for project
### change here to run on cluster ####
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"

# load tokenizer and model
Tokenizer = tokenizer.tokenizer()

# args: lstmLayers, lstmHiddenSize, lstmInputSize, dropout
model = LSTM.LSTM(5, 1000, 1000, 0.3, device).to(device)

# load weights to tokenizer
os.chdir(pathOrigin + "/models")
Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
Tokenizer = Tokenizer.to(device)


# define hyperparameters
params = {"learningRate": 0.001, "weightDecay": 0.01, "epochs": 100, "batchSize": 20, "optimizer": "RMSProp", "validationStep": 1}

# get dataLoaders
#datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "trainData"))
datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

#datasetVal = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "valData"))
datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati")
dataVal = DataLoader(datasetVal, 1, shuffle = True)

# criterion
loss = torch.nn.MSELoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoop(dataTrain, dataVal, Tokenizer,  model, loss, False, "LSTM", params, True, device)




