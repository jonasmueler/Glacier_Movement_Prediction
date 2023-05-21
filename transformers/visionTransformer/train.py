import torch
import functions
import datasetClasses
from torch.utils.data import DataLoader
import visionTransformer
import os 

# device
device = "cuda"
pathOrigin = "/mnt/qb/work/ludwig/lqb875"

# model
model = visionTransformer.visionFuturePrediction(4,50,5, 2, 2, 2500, 2500, 0.5, 0.3, num_classes = 2500).to(device)

# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.0001, "epochs": 30, "batchSize": 100, "optimizer": "adam", "validationStep": 100}

# get dataLoaders /home/jonas/datasets/parbati
datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)

# criterion
loss = torch.nn.MSELoss()

functions.trainLoopUnet(dataTrain, dataVal, model, loss, False, "VisionTransformer", params, True, device)
