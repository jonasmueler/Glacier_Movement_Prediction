import torch
import functions
import datasetClasses
from torch.utils.data import DataLoader
import visionTransformer

# device
device = "cuda"

model = visionTransformer.visionFuturePrediction(4).to(device)


# define hyperparameters
params = {"learningRate": 0.001, "weightDecay": 0.01, "epochs": 5, "batchSize": 100, "optimizer": "adam", "validationStep": 1}

# get dataLoaders /home/jonas/datasets/parbati
datasetTrain = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "train")
#datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

#datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
dataVal = DataLoader(datasetVal, 10, shuffle = True)

# criterion
loss = torch.nn.MSELoss()

functions.trainLoopUnet(dataTrain, dataVal, model, loss, False, "VisionTransformer", params, True, device)