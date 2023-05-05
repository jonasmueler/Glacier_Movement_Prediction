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
