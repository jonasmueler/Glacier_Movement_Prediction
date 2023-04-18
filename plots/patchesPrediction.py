import plottingFunctions
import unet_parts
import unet_model
import os
import datasetClasses
from torch.utils.data import DataLoader
import functions
import torch
import numpy as np
import visionTransformer
import transformerBase
import lstmAttention


# path
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
pathOrigin = "/mnt/qb/work/ludwig/lqb875"

# device
device = "cuda"

# create model
#model = LSTM.LSTM(3,3, 2500, 2500, 0.1, device).to(device)
model = lstmAttention.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)
#model = unet_model.UNet(1,1).to(device)



# load weights
model = plottingFunctions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMAttentionSmall"))
print("loading model finished")



# get dataloader
datasetTest = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "test", bootstrap = False)
#datasetTest = datasetClasses.glaciers("/home/jonas/datasets/parbati", "test", bootstrap = False)
dataTest = DataLoader(datasetTest, 1, shuffle = True)


# first plot patch sequences
nSequences = 10
for i in range(nSequences):
    inpt, target = next(iter(dataTest))
    data = [inpt.to(device).float(), target.to(device).float()]
    plottingFunctions.plotPatches(model, "LSTMAttentionSmall", data, False, device, False)

