import plottingFunctions
import unet_parts
import unet_model
import os
import datasetClasses
from torch.utils.data import DataLoader
import functions
import torch
import numpy as np
import lstmAttention

# path
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
pathOrigin = "/mnt/qb/work/ludwig/lqb875"

# device
device = "cuda"

# create model
#model = unet_model.UNet(1,1).to(device)
model = lstmAttention.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)

# loadWeights
model = plottingFunctions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMAttentionSmall"))
print("loading model finished")


#load scenes
#scenePath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/data/scenes/images"
scenePath = os.path.join(pathOrigin, "datasets", "aletsch", "scenes", "images")
sequences = []
n = 1

# get testset scenes
# get iterator
iterator = np.arange(26 - n*8, 27)
## iterate from back
for i in iterator[0::8][0:-1]:
    inpt = []
    targ = []
    for t in range(4):
        img = functions.openData(os.path.join(scenePath, str(i + t)))
        inpt.append(img)
        target = functions.openData(os.path.join(scenePath, str(i + t + 4)))
        targ.append(target)

    inpt = np.stack(inpt, axis = 0).squeeze()

    targ = np.stack(targ, axis = 0)
    sequences.append([torch.from_numpy(inpt).unsqueeze(dim = 1).to(device).float(),
                      torch.from_numpy(targ).unsqueeze(dim = 1).to(device).float()])




## plot whole scenes
with torch.no_grad():
    for i in range(len(sequences)):
        plottingFunctions.inferenceScenes(model, sequences[i], 50, 50, (1, 600, 600),
                                      "aletsch", str(i), "LSTMAttentionSmall", device,
                                      plot = False, safe = True, pathOrigin = pathOrigin)



