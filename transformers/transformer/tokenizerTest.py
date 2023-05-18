import torch.nn as nn
import torch
from torch.autograd import Variable
import functions
import math
import functions
import numpy as np
import os
from torch.utils.data import DataLoader
import datasetClasses
import torch
import wandb
import matplotlib.pyplot as plt
import tokenizer

Tokenizer = tokenizer.tokenizer()
device = "cpu"
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"

# load weights to tokenizer
os.chdir(pathOrigin + "/models")
Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
Tokenizer = Tokenizer.to(device)
Tokenizer.eval()
# dataloader
tkData = datasetClasses.tokenizerData("/home/jonas/datasets/parbati")
trainLoader = DataLoader(tkData, 1, shuffle = True)

for i in range(5):
    inpt = next(iter(trainLoader))
    inpt = inpt.float()
    plt.imshow(functions.minmaxScaler(inpt.squeeze().detach().numpy()))
    plt.show()
    forward = Tokenizer(inpt)
    forward = torch.reshape(forward, (50, 50))
    plt.imshow(functions.minmaxScaler(forward.detach().numpy()))
    plt.show()

## tokenizer looks shit