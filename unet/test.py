from unet_model import UNet
import torch

import functions
import datasetClasses
from torch.utils.data import DataLoader

# device
device = "cuda"

model = UNet(1,1).to(device)


# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.01, "epochs": 5, "batchSize": 10, "optimizer": "adam", "validationStep": 1}

# get dataLoaders /home/jonas/datasets/parbati
#datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
dataVal = DataLoader(datasetVal, 10, shuffle = True)

# criterion
loss = torch.nn.MSELoss()

functions.trainLoopUnet(dataTrain, dataVal, model, loss, False, "Unet", params, True, device)