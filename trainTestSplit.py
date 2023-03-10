import functions
import torch
import os
import random
import pickle

pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
path = pathOrigin + "/datasets"
os.chdir(path)


# Aletsch glacier
os.chdir(path)
data = functions.openData("aletschStationary")



# get train and test data
crit = 0.8
dTrain = data[0:round(len(data)*crit)]
dValidate = data[round(len(data)*crit):-1]
os.chdir(pathOrigin + "/datasets")

with open("aletschTrainStationary", "wb") as fp:  # Pickling
    pickle.dump(dTrain, fp)
print("data saved!")

with open("aletschValidateStationary", "wb") as fp:  # Pickling
    pickle.dump(dValidate, fp)
print("data saved!")








