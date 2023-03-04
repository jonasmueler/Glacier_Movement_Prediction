import functions
import AuTransformerMaxPool
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt


# global variables
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
#pathOrigin = "/mnt/qb/work/ludwig/lqb875"
device = "cpu"
#device = "cuda"
#load = True
load = False

# load model
model = AuTransformerMaxPool.AE_Transformer(9680, 100, 100, 1, 1, 1, 1, 1,torch.device(device), True, 5)
model = model.to(torch.device(device)).to(torch.float32)

if load == True:
    path = pathOrigin + "/results/hardConditionedOutputSoftmax/hardConditionedOutputSoftmax"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)


## load dataset

#os.chdir(pathOrigin + "/datasets")
#aletschData = functions.openData("aletsch_patches")

### debug ###
# test with 200*200 image
testInput = torch.rand(5, 3, 50, 50, requires_grad=True)
testTargets = torch.rand(5, 1, 50, 50, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])


aletschData = [[[testInput, testInputDates], [testTargets, testTargetDates]],
        [[testInput, testInputDates], [testTargets, testTargetDates]],
        [[testInput, testInputDates], [testTargets, testTargetDates]]]



# args: model, data, path, plot
functions.plotPatches(model, aletschData[0], pathOrigin + "/predictions", False)


"""

testInput = torch.rand(5, 3, 50, 50, requires_grad=True)
testTargets = torch.rand(5, 1, 50, 50, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])


sceneDataHelheim = [[[testInput, testInputDates], [testTargets, testTargetDates]]]


for i in range(20):
    functions.inferenceScenes(model,
                                sceneDataHelheim[i],
                                50,
                                50,
                                (1,300,300),
                                "Helheim",
                                str(i),
                                "transformerScenesSmall",
                                device,
                                plot = True,
                                safe = True)
"""