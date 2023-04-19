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
import ConvLSTM


# path
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"

# device
device = "cpu"

# create model
#model = visionTransformer.visionFuturePrediction(4,50,5, 2, 2, 800, 800, 0.1, 0.1, num_classes = 2500).to(device)
#model = unet_model.UNet(1,1).to(device)
#model = transformerBase.Transformer(2500, 1, 1, device, predictionInterval=4).to(device)
# load weights
#model = plottingFunctions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "Unet"))
model = ConvLSTM.ConvLSTMPredictor([2, 2, 2, 2, 2, 2]).to(device)
print("loading model finished")



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
    plottingFunctions.plotPatches(model, "Unet", data, False, device, False)

"""
#load scenes
scenePath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/data/scenes/images"
#scenePath = os.path.join(pathOrigin, "datasets", "parbati", "scenes")
sequences = []
n = 1

# get testset scenes
# get iterator
iterator = np.arange(35 - n*8, 36)
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
        plottingFunctions.inferenceScenes(model, sequences[i], 50, 50, (1, 800, 800),
                                      "parvati", str(i), "ConvLSTM", device,
                                      plot = True, safe = True, pathOrigin = pathOrigin)


