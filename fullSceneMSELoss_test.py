import functions
import models
import torch
from torchvision import transforms
from PIL import Image

model = models.AE_Transformer(2420,2420, 2420, 3, 1, 1000, 1,1, True, None)
# test with 200*200 image
testInput = torch.rand(5, 3, 200, 200, requires_grad=True)
testTargets = torch.rand(5, 1, 200, 200, requires_grad=True)

testInputDates = [torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)]

testTargetDates = [torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)]

## final test of functions

#res = functions.fullSceneLoss(testInput, testInputDates, testTargets, testTargetDates, model, 50, 40, (1,200,200))

#print(res)

## test pipeline without model predictions and with same images as targets, the loss should be 0

path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/Helheim/filename.jpg"
img = Image.open(path)
convert_tensor = transforms.ToTensor()
t = convert_tensor(img)[:, 0:200, 0:200]
l = [t,t,t,t,t]
l = torch.stack(l, dim = 0)

target = convert_tensor(img)[0, 0:200, 0:200]
b = [target, target, target, target, target]
b = torch.stack(b, dim = 0)
b = b.unsqueeze(dim = 1)

#res = functions.fullSceneLoss(l, testInputDates, l, testTargetDates, model, 50, 40, (3,200,200), test = True)
#print(res)

d = [[l, testInputDates], [b, testTargetDates]]

functions.inferenceScenes(model, d, 50, 50, (1,200,200), plot = True)

