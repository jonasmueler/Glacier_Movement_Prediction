import pickle
import functions
import models
import random
import torch
import os
import torch.optim as optim

## load datasets of three glaciers
"""
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
"""
### debug ###
# test with 200*200 image
testInput = torch.rand(5, 3, 50, 50, requires_grad=True)
testTargets = torch.rand(5, 1, 50, 50, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])


data = [[[testInput, testInputDates], [testTargets, testTargetDates]],
        [[testInput, testInputDates], [testTargets, testTargetDates]],
        [[testInput, testInputDates], [testTargets, testTargetDates]]]

############

# random shuffle
random.shuffle(data)

# get train and test data
crit = 0.9
dTrain = data[0:round(len(data)*crit)]
dValidate = data[round(len(data)*crit):-1]

# move to cuda
dTrain = list(map(lambda x: functions.moveToCuda(x, torch.device('cpu')), dTrain))
dValidate = list(map(lambda x: functions.moveToCuda(x, torch.device('cpu')), dValidate))

# initialize model
model = models.AE_Transformer(2420,2420,2420, 1, 1, 1, 1, 1,torch.device('cpu'), True, 5)
model = model.to(torch.device('cpu')).to(torch.float32)
"""
# train on patches
### args ### data, model, loadModel, modelName, lr, weightDecay, earlyStopping, epochs, validationSet, validationStep
functions.trainLoop(dTrain, model, False,"transformerPatches", 0.0001, 0.01, 0.00001, 1, dValidate, 10)

# load full scene dataset, use Helheim data to train edges between patch predictions
"""
"""
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim"
os.chdir(path)
sceneDataHelheim = functions.loadFullSceneData(path,
                                               ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"],
                                               5,
                                               [7, 8, 9],
                                               9,
                                               [100, 400, 200, 500])
"""
### debug ###
testInput = torch.rand(5, 3, 300, 300, requires_grad=True)
testTargets = torch.rand(5, 1, 300, 300, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])


sceneDataHelheim = [[[testInput, testInputDates], [testTargets, testTargetDates]]]

#############
"""
# train on full scene dataset
functions.fullSceneTrain(model, "transformerScenes", optim.Adam(model.parameters(), lr=0.0001, weight_decay= 0.01),
                                 sceneDataHelheim,
                                 1,
                                 50, 50,
                                 (1, 300 ,300))
"""
## predict some images and save them on harddrive
## args: model, data, patchSize, stride, outputDimensions, glacierName, predictionName, modelName, plot = False, safe = False
res1 = functions.inferenceScenes(model,
                                sceneDataHelheim[0],
                                50,
                                50,
                                (1,200,200),
                                "Helheim",
                                "0",
                                "transformerScenes",
                                plot = False,
                                safe = True)
"""
res2 = functions.inferenceScenes(model,
                                sceneDataHelheim[1],
                                50,
                                50,
                                (1,200,200),
                                "Helheim",
                                "1",
                                plot = False,
                                safe = True)

res3 = functions.inferenceScenes(model,
                                sceneDataHelheim[2],
                                50,
                                50,
                                (1,200,200),
                                "Helheim",
                                "2",
                                plot = False,
                                safe = True)
                                
"""

## integrate weights and biases into the script to observe losses





