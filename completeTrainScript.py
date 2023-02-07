import pickle
import functions
import models
import random
import torch


## load datasets of three glaciers

# helheim glacier
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim/patched"
os.chdir(path)
helheim = functions.openData("trainData")

# Aletsch glacier
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn/patched"
os.chdir(path)
aletsch = functions.openData("trainData")

# Jakobshavn glacier
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jakobshavn/patched"
os.chdir(path)
jakobshavn = functions.openData("trainData")

# put into one big dataset
data = helheim + aletsch + jakobshavn

# random shuffle
random.shuffle(data)

# get train and test data
crit = 0.9
dTrain = data[0:round(len(d)*crit)]
dValidate = data[round(len(d)*crit):-1]

# move to cuda
dTrain = list(map(lambda x: functions.moveToCuda(x), dTrain))
dValidate = list(map(lambda x: functions.moveToCuda(x), dValidate))

# initialize model
model = models.AE_Transformer(2420,2420,2420, 3, 5, 1000, 12, 10, True, None)
model = model.to("cuda").to(torch.float32)

# train on patches
### args ### data, model, loadModel, modelName, lr, weightDecay, earlyStopping, epochs, validationSet, validationStep
functions.trainLoop(dTrain, model, False,"transfomrerPatches", 0.0001, 0.01, 0.00001, 1000, dValidate)

## to do: load full scene dataset
## train on full scene dataset
## get losses from function
## predict some images and save them on harddrive
## integrate weights and biases into the script to observe losses





