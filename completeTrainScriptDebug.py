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
import TransformerSoftConditioning
import TransformerTemporal
import TransformerStationary
import pureTransformer

## load datasets of three glaciers

## global variables for project
### change here to run on cluster ####
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"


# helheim glacier
path = pathOrigin + "/datasets"
#os.chdir(path)
#helheim = functions.openData("trainDataHelheim")

# Aletsch glacier
os.chdir(path)
aletsch = functions.openData("aletschValidateStationary")

# Jakobshavn glacier
#os.chdir(path)
#jakobshavn = functions.openData("trainDataJakobshavn")

# put into one big dataset
data = aletsch #helheim + aletsch + jakobshavn

"""
### debug ###
# test with 200*200 image
testInput = torch.rand(4, 3, 50, 50, requires_grad=True)
testTargets = torch.rand(4, 1, 50, 50, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True)])


data = [[[testInput, testInputDates], [testTargets, testTargetDates]],
        [[testInput, testInputDates], [testTargets, testTargetDates]],
        [[testInput, testInputDates], [testTargets, testTargetDates]]]
"""
############

# random shuffle
random.shuffle(data)

# get train and test data
crit = 0.9
dTrain = data[0:round(len(data)*crit)]
dValidate = data[round(len(data)*crit):-1]

# move to cuda
#dTrain = list(map(lambda x: functions.moveToCuda(x, torch.device(device)), dTrain))
#dValidate = list(map(lambda x: functions.moveToCuda(x, torch.device(device)), dValidate))

# initialize model
## args ## encoderIn, hiddenLenc, hiddenLdec, mlpSize, numLayersDateEncoder, sizeDateEncoder,
# attLayers, attentionHeads, device, Training=True, predictionInterval=None
#model = models.AE_Transformer(2420,2420,2420, 3, 2, 1000, 10, 10,torch.device('cuda'), True, 5)

model = pureTransformer.AE_Transformer(2500,1000,1000, 1, 1, 1, 1, 1,torch.device(device), True, 4)

model = model.to(torch.device(device)).to(torch.float32)

# train on patches
### args ### (data, model, loadModel, modelName, lr, weightDecay, earlyStopping, epochs,
              # validationSet, validationStep, WandB, device, pathOrigin = pathOrigin):
functions.trainLoop(dTrain, model, False,"transformerPatches", 0.0001, 0.01, 0, 2, None, 1, True, device, 4)
#functions.trainLoop(dTrain, model, True,"hardConditionedOutputSoftmax", 0.00001, 0.01, 0.00001, 1, dValidate, 10, True, device)
# load full scene dataset, use Helheim data to train edges between patch predictions


#path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim"
#os.chdir(path)
#sceneDataHelheim = functions.loadFullSceneData(path,
#                                               ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"],
#                                               5,
#                                               [7, 8, 9],
#                                               9,
#                                               [100, 400, 200, 500])
"""
scenedataHelheim = functions.openData("trainDataFullScenes")
sceneDataHelheim = list(map(lambda x: functions.moveToCuda(x, torch.device(device)), sceneDataHelheim))
"""
### debug ###
testInput = torch.rand(4, 3, 300, 300, requires_grad=True)
testTargets = torch.rand(4, 1, 300, 300, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True)])


sceneDataHelheim = [[[testInput, testInputDates], [testTargets, testTargetDates]]]

#############

# train on full scene dataset
# model, modelName, optimizer, data, epochs, patchSize, stride, outputDimensions, device,
#                    WandB, pathOrigin = pathOrigin
functions.fullSceneTrain(model, "transformerScenes", optim.Adam(model.parameters(), lr=0.0001, weight_decay= 0.01),
                                 sceneDataHelheim,
                                 1,
                                 50, 50,
                                 (1, 300 ,300),
                                device,
                                True)

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
                                device,
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







