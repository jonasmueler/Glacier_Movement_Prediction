import functions
import torch
import torch.nn as nn

# load data
# targets
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/experiment3/results/LSTMAttentionSmallMixedData/modelPredictions/aletsch/0/targets"

# Load the two PDF files as images
images1 = functions.openData(path + "/0")[0,:,:]
images2 = functions.openData(path + "/1")[0,:,:]
images3 = functions.openData(path + "/2")[0,:,:]
images4 = functions.openData(path + "/3")[0,:,:]

targets = [torch.from_numpy(img) for img in [images1, images2, images3, images4]]
targets = torch.stack(targets)

# LSTMattentionsmall
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/experiment3/results/LSTMAttentionSmallMixedData/modelPredictions/aletsch/0/predictions"
# Load the two PDF files as images
images5 = functions.openData(path + "/0")[0,:,:]
images6 = functions.openData(path + "/1")[0,:,:]
images7 = functions.openData(path + "/2")[0,:,:]
images8 = functions.openData(path + "/3")[0,:,:]

train = [torch.from_numpy(img) for img in [images5, images6, images7, images8]]
train = torch.stack(train)


MSE = nn.MSELoss()
MAE = nn.L1Loss()

print(MSE(targets, train))
print(MAE(train, targets))

