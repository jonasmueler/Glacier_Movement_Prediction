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


# path
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"

# device
device = "cuda"

# create model
model = visionTransformer.visionFuturePrediction(4,50,5, 2, 2, 800, 800, 0.1, 0.1, num_classes = 2500).to(device)
#model = unet_model.UNet(1,1).to(device)
# load weights
#model = plottingFunctions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "VisionTransformer"))
#print("loading model finished")



# get dataloader
#datasetTest = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "test", bootstrap = False)
#datasetTest = datasetClasses.glaciers("/home/jonas/datasets/parbati", "test", bootstrap = False)
#dataTest = DataLoader(datasetTest, 1, shuffle = True)
"""
# first plot patch sequences
nSequences = 5
for i in range(nSequences):
    inpt, target = next(iter(dataTest))
    data = [inpt.to(device).float(), target.to(device).float()]
    plottingFunctions.plotPatches(model, data, False, device, True)

"""
#load scenes
scenePath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/parbatiScenes"
sequences = []
n = 3
for i in range(n):
    inpt = []
    targ = []
    for t in range(4):
        img = functions.openData(os.path.join(scenePath, str(i + t)))
        inpt.append(img)
        target = functions.openData(os.path.join(scenePath, str(i + t + 4)))
        targ.append(target)

    inpt = np.stack(inpt, axis = 0)
    targ = np.stack(targ, axis = 0)
    sequences.append([torch.from_numpy(inpt).unsqueeze(dim = 1).to(device).float(),
                      torch.from_numpy(targ).unsqueeze(dim = 1).to(device).float()])


## plot whole scenes
with torch.no_grad():
    for i in range(len(sequences)):
        plottingFunctions.inferenceScenes(model, sequences[i], 50, 50, (1, 800,800),
                                      "parvati", str(i), "Unet2", device,
                                      plot = True, safe = True, pathOrigin = pathOrigin)


