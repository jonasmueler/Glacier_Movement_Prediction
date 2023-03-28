import pickle
import functions
import models
import random
import torch
import os
import torch.optim as optim
import AuTransformerMaxPool
import TransformerNoEmbedding
import Transformer

## global variables for project
### change here to run on cluster ####
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"

pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"


# model
model = Transformer.AE_Transformer(2500,2500,2500, 1, 1, 1, 1, 1,torch.device(device), True, 5)
model = model.to(torch.device(device)).to(torch.float32)

# data
path = pathOrigin + "/datasets"
os.chdir(path)
sceneDataHelheim = functions.openData("trainDataFullScenes")
sceneDataHelheim = list(map(lambda x: functions.moveToCuda(x, torch.device(device)), sceneDataHelheim))
print(sceneDataHelheim[0][0][0].size())
print("data loading finished")

# train on full scene dataset
functions.fullSceneTrain(model, "transformerScenes", optim.Adam(model.parameters(), lr=0.0001, weight_decay= 0.01),
                                 sceneDataHelheim,
                                 1,
                                 50, 50,
                                 (1, 300 ,300),
                                device, True)