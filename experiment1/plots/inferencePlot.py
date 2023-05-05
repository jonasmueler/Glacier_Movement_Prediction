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
import LSTM
import lstmAttention
import ConvLSTM
import transformerLSTM

# path
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
pathOrigin = "/mnt/qb/work/ludwig/lqb875"

# device
device = "cuda"

#model 
modelName = "LSTMAttentionSmall"

# create model
#model = unet_model.UNet(1,1).to(device)
#model = LSTM.LSTM(3,3, 2500, 2500, 0.1, device).to(device)
model = lstmAttention.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)
#model = ConvLSTM.ConvLSTMPredictor([64, 64, 24, 24, 64, 24]).to(device)


# load weights to transformers
model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMAttentionSmall"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "Unet"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "ConvLSTM"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMEncDec"))
print("loading models finished")




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
#scenePath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images"
scenePath = os.path.join(pathOrigin, "datasets", "parbati", "scenes", "images")
sequences = []
n = 1 # number of predictions

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

    inpt = np.stack(inpt, axis = 0)
    targ = np.stack(targ, axis = 0)
    sequences.append([torch.from_numpy(inpt).unsqueeze(dim = 1).to(device).float(),
                      torch.from_numpy(targ).unsqueeze(dim = 1).to(device).float()])


## plot whole scenes
with torch.no_grad():
    for i in range(len(sequences)):
        plottingFunctions.inferenceScenes(model, sequences[i], 50, 50, (1, 800, 800),
                                      "parvati", str(i), modelName, device,
                                      plot = True, safe = True, pathOrigin = pathOrigin)


