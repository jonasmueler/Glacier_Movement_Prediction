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

# get data onto cuda
def helper(y):
    """
    transfers datum to gpu

    y: list of list of tensor and tensor and list of tensor and tensor
        input datum
    return: list of list of tensor and tensor and list of tensor and tensor
        transferred to cuda gpu
    """
    y[0][0] = y[0][0].to("cuda").to(torch.float32)
    y[0][1] = y[0][1].to("cuda").to(torch.float32)
    y[1][0] = y[1][0].to("cuda").to(torch.float32)
    y[1][1] = y[1][1].to("cuda").to(torch.float32)

    return y


dTrain = list(map(lambda x: helper(x), dTrain))
dTest = list(map(lambda x: helper(x), dValidate))


# initialize model
model = models.AE_Transformer(2420,2420,2420, 3, 5, 1000, 12, 10, True, None)
model = model.to("cuda").to(torch.float32)





