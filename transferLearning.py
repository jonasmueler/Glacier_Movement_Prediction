import torch
import functions
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image


## test image functions
#path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/Helheim/filename.jpg"
#img = Image.open(path)

#convert_tensor = transforms.ToTensor()

#tensor = convert_tensor(img)



def getPatchesTransfer(tensor, patchSize, stride=1):

    """
    takes an image and outputs list of patches in the image

    tensor: tensor
        input image
    patchSize: int
        x,y dimension of patch
    stride: int
        if bigger than 1 patches overlap

    returns: list of tensor
        list of patches
    """
    # Get image dimensions
    nChannels, height, width = tensor.shape
    # Calculate the number of patches in each direction
    nHorizontalPatches = (width - patchSize) // stride + 1
    nVerticalPatches = (height - patchSize) // stride + 1
    # Initialize a list to store the patches
    patches = []
    # Iterate over the patches and extract them
    for i in range(nVerticalPatches):
        for j in range(nHorizontalPatches):
            # Calculate the top-left corner of the patch
            x = j * stride
            y = i * stride
            # Extract the patch
            patch = tensor[:, y:y+patchSize, x:x+patchSize]
            # Add the patch to the list
            patches.append(patch)
    # Return the list of patches
    return patches


def combinePatchesTransfer(patches, tensorShape, patchSize, stride=1):

    """
    combines a list of patches to full image

    patches: list of tensor
        patches in list
    tensorShape: tuple
        shape of output tensor
    patchSize: int
        x,y
    stride: int

    returns: tensor
        image in tensor reconstructed from the patches
    """
    # Get the number of channels and the target height and width
    n_channels, height, width = tensorShape
    # Initialize a tensor to store the image
    tensor = torch.zeros(tensorShape)
    # Calculate the number of patches in each direction
    nHorizontalPatches = (width - patchSize) // stride + 1
    nVerticalPatches = (height - patchSize) // stride + 1
    # Iterate over the patches and combine them
    patchIndex = 0
    for i in range(nVerticalPatches):
        for j in range(nHorizontalPatches):
            # Calculate the top-left corner of the patch
            x = j * stride
            y = i * stride
            # Get the patch and add it to the image
            patch = patches[patch_index]
            tensor[:, y:y+patchSize, x:x+patchSize] += patch
            patchIndex += 1
    # Return the image tensor
    return tensor

# input 5, 3, 50, 50; targets: 5, 3, 50, 50
def fullSceneLoss(inputScenes, inputDates, targetScenes, targetDates, model):
    """

    inputScenes: tensor
        scenes
    inputDates: tensor
        dates
    targetScenes: tensor
        target scenes
    targetDates: tensor
        target dates
    model: torch.model object

    returns: int
        loss on full five scenes and all associated patches

    """

    # get patches from input images and targets
    inputList = []
    targetList = []
    for i in range(len(inputScenes)):
        helper = getPatchesTransfer(inputScenes[i], 50)
        inputList.append(helper)

        helper = getPatchesTransfer(targetScenes[i], 50)
        targetList.append(helper)

    # get predictions from input patches
    latentSpaceLoss = 0
    for i in range(len(inputList[0])):
        helperInpt = [x[i] for x in inputList]
        targetInpt = [x[i] for x in targetList]
        inputPatches = torch.stack(helperInpt, dim = 0)
        targetPatches = torch.stack(targetInpt, dim=0)

        # put together for final input
        finalInpt = [[inputPatches, inputDates], [targetPatches, targetDates]]

        # predict with model
        prediction = model.forward(finalInpt, training = True)

        # switch input with predictions; z = scene index, i = patch index
        for z in range(len(prediction[0])):
            inputList[z][i] = prediction[0][z, :, :]

        # accumulate latent space losses
        latentSpaceLoss += prediction[1].item()

    # get final loss of predictions of the full scenes
    # set patches back to images
    scenePredictions = []
    for x in range(inputList):
        scene = combinePatchesTransfer(x)
        scenePredictions.append(scene)

    fullLoss = sum(list(map(lambda x,y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))

    fullLoss += latentSpaceLoss

    return fullLoss







