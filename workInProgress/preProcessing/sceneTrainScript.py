import functions
import torch
import torch.optim as optim
import os
import AuTransformerMaxPool

pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cpu"

model = AuTransformerMaxPool.AE_Transformer(9680, 10, 10, 1, 1, 1, 1, 1,torch.device(device), True, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)
"""
#path = pathOrigin + "/conditionedOutputSoftmax/conditionedOutputSoftmax"
#checkpoint = torch.load(path)

#model.load_state_dict(checkpoint['state_dict'])
"""
# load dataset
os.chdir(pathOrigin + "/datasets")
sceneDataAletsch = functions.openData("test")
for i in range(len(sceneDataAletsch)):
    print(torch.nonzero(torch.isnan(sceneDataAletsch[i][0][1])))
    print(torch.nonzero(torch.isnan(sceneDataAletsch[i][1][1])))

#print(sceneDataAletsch)
# move to cuda
#sceneDataAletsch = list(map(lambda x: functions.moveToCuda(x, torch.device(device)), sceneDataAletsch))


# train on scenes
functions.fullSceneTrain(model, "transformerScenes", optimizer,
                                 sceneDataAletsch,
                                 3,
                                 50, 50,
                                 (1, 600 ,600),
                                device,
                                True)
