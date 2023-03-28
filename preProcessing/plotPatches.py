import pickle
import pandas as pd
import functions
import random
import torch
import os
import torch.optim as optim
import transformerBase
import tokenizer
from torch.utils.data import DataLoader
import datasetClasses
import numpy as np
import matplotlib.pyplot as plt


# load model

## global variables for project
### change here to run on cluster ####
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"

# load tokenizer and model
Tokenizer = tokenizer.tokenizer()


## args: hiddenLenc, attLayers, attentionHeads, device, Training=True, predictionInterval=None)
transformerSmall = transformerBase.Transformer(1000, 2, 4, device, predictionInterval=4).to(device)
#transformerMiddle = transformerBase.Transformer(1000, 4, 8, device, predictionInterval=4).to(device)
#transformerBig = transformerBase.Transformer(1000, 6, 10, device, predictionInterval=4).to(device)


# load weights to transformers
transformerSmall = functions.loadCheckpoint(transformerSmall, None, os.path.join(pathOrigin, "models", "TransformerSmall"))
#transformerMiddle = functions.loadCheckpoint(transformerMiddle, None, os.path.join(pathOrigin, "models", "TransformerIntermediate"))
#transformerBig = functions.loadCheckpoint(transformerBig, None, os.path.join(pathOrigin, "models", "TransformerBig"))


# load weights to tokenizer
os.chdir(pathOrigin + "/models")
Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
Tokenizer = Tokenizer.to(device)


print("loading models finished")

# dataLoader /home/jonas/datasets/parbati
#datasetVal = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "test")
datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "test")
dataTest = DataLoader(datasetVal, 1, shuffle = True)


for i in range(5):
    x,y = next(iter(dataTest))
    functions.plotPatches(transformerSmall, [x,y], True, Tokenizer, device, True)