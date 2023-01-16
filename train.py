import functions
import torch
import models
# data
d = functions.loadData("D:/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/Helheim",
                       ["trainData"])

#d = functions.loadData("D:/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/Helheim",
#                       ["trainData"])
crit = 0.90
dTrain = d[0:round(len(d)*crit)]
dTest = d[round(len(d)*crit):-1]

# model
#model = models.VAE_LSTM(484, 10, 10, 10, 10, 484, 484)
#model = models.VAE_LSTM(484, 10, 10, 10, 10, 484, 484) # works reasonably good
#model = models.VAE_LSTM(484, 10, 5, 5, 10, 484, 484) # not as good
model = models.AE_Transformer(2420,2420, 2420, 3, 1, 1000, 6,4, True, None)
model = model.to(torch.float)

res = functions.trainLoop(dTrain, model, False,"Transformer.pth.tar", 0.01, 0.01, -1, functions.MSEpixelLoss, 1000, 2, dTest)

#hiddenL, mlpSize, numLayersDateEncoder, sizeDateEncoder, lstmLayers, lstmHiddenSize, lstmInputSize