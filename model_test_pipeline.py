import functions
import TransformerNoEmbedding
import torch
import torch.optim as optim
import os

pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"

model = TransformerNoEmbedding.AE_Transformer(484,484,484, 1, 1, 480, 11, 22,torch.device(device), True, 5)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/models/transformerPatchesSmall/transformerPatchesSmall"
checkpoint = torch.load(path)

model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(pytorch_total_params)

## get predictions

os.chdir(pathOrigin + "/datasets")
sceneDataHelheim = functions.openData("trainDataFullScenes")
"""

testInput = torch.rand(5, 3, 300, 300, requires_grad=True)
testTargets = torch.rand(5, 1, 300, 300, requires_grad=True)

testInputDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])

testTargetDates = torch.stack([torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True),
                    torch.rand(1,3, requires_grad=True), torch.rand(1,3, requires_grad=True)])


sceneDataHelheim = [[[testInput, testInputDates], [testTargets, testTargetDates]]]
"""

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





