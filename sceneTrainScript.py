import functions
import TransformerNoEmbedding
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt

pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"

model = TransformerNoEmbedding.AE_Transformer(484,484,484, 1, 1, 480, 11, 22,torch.device(device), True, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)


path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/models/transformerPatchesSmall/transformerPatchesSmall"
checkpoint = torch.load(path)

model.load_state_dict(checkpoint['state_dict'])

# load dataset
os.chdir(pathOrigin + "/datasets")
sceneDataHelheim = functions.openData("aletschFullScenes")