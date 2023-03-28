import functions
import models
import torch
from torchvision import transforms
from PIL import Image
import torch.optim as optim

model = models.AE_Transformer(2420,2420, 2420, 3, 1, 1000, 1,1, True, 5)

# test with 200*200 image
testInput = torch.rand(5, 3, 300, 300, requires_grad=True)
testTargets = torch.rand(5, 1, 300, 300, requires_grad=True)

testInputDates = [torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)]

testTargetDates = [torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)]

## final test of functions

#res = functions.fullSceneLoss(testInput, testInputDates, testTargets, testTargetDates, model, 50, 50, (1,200,200), training = True, test = False)

#print(res)

d = [[[testInput, testInputDates], [testTargets, testTargetDates]]]
train = functions.fullSceneTrain(model, "model", optim.Adam(model.parameters(), lr=0.0001, weight_decay= 0.01),
                                 d,
                                 3,
                                 50, 50,
                                 (1, 300 ,300))
print(train)

## test pipeline without model predictions and with same images as targets, the loss should be 0
"""
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/Helheim/filename.jpg"
img = Image.open(path)
convert_tensor = transforms.ToTensor()
t = convert_tensor(img)[:, 0:200, 0:200]
l = [t,t,t,t,t]
l = torch.stack(l, dim = 0)
print(l.size())
target = convert_tensor(img)[0, 0:200, 0:200]
b = [target, target, target, target, target]
b = torch.stack(b, dim = 0)
b = b.unsqueeze(dim = 1)


res = functions.fullSceneLoss(l, testInputDates, l, testTargetDates, model, 50, 40, (1,200,200), training=True, test = False)
print(res)

#d = [[l, testInputDates], [b, testTargetDates]]

#functions.inferenceScenes(model, d, 50, 50, (1,200,200), plot = True)

"""