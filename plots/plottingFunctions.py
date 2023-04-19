import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle

pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"

def minmaxConvertToRGB(X, ret=False):
    """
    X: 2d array
    returns: 2d array
       values from 0-255
    """
    if ret == False:
        res = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
        res = res * 255.999
        return res
    elif ret:
        res = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
        res = res * 255.999
        return [res, np.nanmin(X), np.nanmax(X)]

def minmaxScaler(X):
    """
    X: 2d array
    returns: 2d array
       values from [0,1]
    """
    #res = (X - np.nanpercentile(X,2)) / (np.nanpercentile(X, 98) - np.nanpercentile(X, 2))
    res = ((X - np.nanmin(X) )/ (np.nanmax(X) - np.nanmin(X)))

    return res


def createImage(img, alpha):
    """
    img: 3d array
        [red, green, blue]


    returns: np.array
        plot ready image
    """
    red = img[0, :,:]*alpha
    green = img[1, :,:]*alpha
    blue = img[2, :,:]*alpha

    green = minmaxScaler(green)
    blue = minmaxScaler(blue)
    red = minmaxScaler(red)

    plotData = np.dstack((red, green, blue))

    return plotData

# kernel
def kernel(x, mask):
    """
    x: np.array

    returns: float
        applied average kernel on one of the nan pixels
    """
    # Kernel from Maria-Minerva Vonica, Romania; Andrei Ancuta; Marc Frincu (2021)

    kernelMask = np.array([[1, 2, 3, 2, 1],
                           [2, 4, 6, 4, 2],
                           [3, 6, 9, 6, 3],
                           [2, 4, 6, 4, 2],
                           [1, 2, 3, 2, 1]])

    # get final kernel
    k = np.multiply(kernelMask, mask)

    # calculate weighted average
    res = np.ma.average(np.nan_to_num(np.ndarray.flatten(x)), weights=np.ndarray.flatten(k))

    return res


def applyToImage(img):
    """
    img: 2d np.array

    returns: 2d np.array

        array with imputed missings

    """

    # create matrix, where value is missing matrix is 0 otherwise 1
    missings = np.argwhere(np.isnan(img))
    zer = np.ones(img.shape)
    for i in range(len(missings)):
        zer[missings[i][0], missings[i][1]] = 0
        missings[i] = missings[i] + 2

    ## add 0 padding
    zer = np.vstack([np.zeros((2, len(zer[0, :]))), zer, np.zeros((2, len(zer[0, :])))])
    zer = np.hstack([np.zeros((len(zer[:, 0]), 2)), zer, np.zeros((len(zer[:, 0]), 2))])

    img = np.vstack([np.zeros((2, len(img[0, :]))), img, np.zeros((2, len(img[0, :])))])
    img = np.hstack([np.zeros((len(img[:, 0]), 2)), img, np.zeros((len(img[:, 0]), 2))])

    for i in range(len(missings)):
        # calculate value with kernel
        patch = img[missings[i][0] - 2:(missings[i][0] - 2) + 5, missings[i][1] - 2:(missings[i][1] - 2) + 5]
        mask = zer[missings[i][0] - 2:(missings[i][0] - 2) + 5, missings[i][1] - 2:(missings[i][1] - 2) + 5]
        res = kernel(patch, mask)
        img[missings[i][0], missings[i][1]] = res

    return img[2:-2, 2:-2]


# create mean image (unaligned) and use it for the imputation of remainig missnig values in the satellite image
def imputeMeanValues(d, band):
    """
    creates mean image nad imputes values for areas which are not covered from the satelite

    d: list of tuple of datetime and ndarray
    bands: bands to be averaged over

    returns: list of tuple of datetime and ndarray
        with imputed values over the edges
    """
    # get images without missing corners
    idx = []
    idxMissing = []
    for i in range(len(d)):
        if np.sum(np.isnan(d[i][1][band, :, :])) == 0:
            idx.append(i)
        if np.sum(np.isnan(d[i][1][band, :, :])) > 0:
            idxMissing.append(i)

    Mean = d[idx[0]][1][band, :, :]
    for i in idx[1:]:
        Mean += d[i][1][band, :, :]

    Mean = Mean / len(idx)
    # impute mean values into images with missing corners
    for z in range(len(idxMissing)):
        img = d[idxMissing[z]][1][band, :, :]
        missings = np.argwhere(np.isnan(img))

        for x in range(len(missings)):
            insert = Mean[missings[x][0], missings[x][1]]
            img[missings[x][0], missings[x][1]] = insert
        d[idxMissing[z]][1][band, :, :] = img

    return d

def getPatches(tensor, patchSize, stride=50):

    """
    takes an image and outputs list of patches in the image

    tensor: tensor
        input image
    patchSize: int
        x,y dimension of patch
    stride: int

    returns: list of tensor
        list of patches
    """
    # Get image dimensions
    nChannels, height, width = tensor.shape
    # Calculate the number of patches in each direction
    nHorizontalPatches = (width - patchSize) // stride + 1
    nVerticalPatches = (height - patchSize) // stride + 1

    # Iterate over the patches and extract them
    patches = []
    counterX = 0
    counterY = 0
    for i in range(nVerticalPatches):
        for j in range(nHorizontalPatches):
            patch = tensor[:, counterX:counterX + patchSize, counterY:counterY + patchSize]
            # update counters
            counterX += stride

            # Add the patch to the list
            patches.append(patch)
        counterY += stride
        counterX = 0
    return patches


def combinePatches(patches, tensorShape, patchSize, stride=50, device= "cpu"):

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
    tensor = torch.zeros(tensorShape).to(device)

    # Calculate the number of patches in each direction
    nHorizontalPatches = (width - patchSize) // stride + 1
    nVerticalPatches = (height - patchSize) // stride + 1

    # Iterate over the patches and combine them
    patchIndex = 0
    counterX = 0
    counterY = 0
    for i in range(nVerticalPatches):
        for j in range(nHorizontalPatches):
            tensor[:, counterX:counterX + patchSize, counterY:counterY + patchSize] = patches[patchIndex]

            # update counters
            counterX += stride
            patchIndex += 1

        counterY += stride
        counterX = 0

    return tensor

"""
## test image functions
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/Helheim/filename.jpg"
img = Image.open(path)
convert_tensor = transforms.ToTensor()
t = convert_tensor(img)[:, 0:200, 0:200]
t_original = convert_tensor(img)[:, 0:200, 0:200]
print(t.size())
plt.imshow(np.transpose(t.numpy(), (1,2,0)))
plt.show()

t = getPatchesTransfer(t, 50, 40)
plt.imshow(np.transpose(t[0].numpy(), (1,2,0)))
plt.show()
print(len(t))
t = combinePatchesTransfer(t, (3, 200,200), 50, 40)
plt.imshow(np.transpose(t.numpy(), (1,2,0)))
plt.show()

print(t_original.numpy()- t.numpy())
"""

def createPatches(img, patchSize, stride):
    """
    creates image patches sampled from a region of interest

    img: np.array
    patchSize: int
        size of patches
    stride: int

    returns:  torch.tensor
        shape = (n_patches, bands, patchSize, patchSize)
    """
    # torch conversion, put into ndarray
    img = torch.from_numpy(img)
    patches = getPatches(img, patchSize, stride=stride)
    out = torch.stack(patches, dim = 0)
    out = out.numpy()

    return out


def automatePatching(data, patchSize, stride):
    """
    creates image patches sampled from a region of interest

    data: list of tuple of datetime and np.array
        data extracted from API
    patchSize: int
        size of patches
    maxPatches: int
        number of patches extracted

    returns:  list of tuple of datetime and np.array
        switch np array in tuple with np array with on more dimension -> patches
    """

    res = []
    for i in range(len(data)):
        print("patchify scene: ", i)
        patches = createPatches(np.expand_dims(data[i], axis=0), patchSize, stride)
        res.append(patches)

    return res


## visualize network performance on full scenes, use for testData, qualitative check
def inferenceScenes(model, data, patchSize, stride, outputDimensions, glacierName, predictionName, modelName, device, plot = False, safe = False, pathOrigin = pathOrigin):
    """
    use for visual check of model performance

    model: nn.class object
    data: same as above
    patchSize: int
    stride: int
    outputDimensions: tuple
    glacierName: str
        name of the glacier for order structure
    predictionname: str
        name of folder for predictions to be safed in
    modelName: string
        name of the model to safe in order structure
    device: sring
        machine to compute on
    plot: boolean
    safe: boolean
        safe output as images on harddrive

    return: list of tensor
        predicted scenes
    """
    # inference mode
    model.eval()

    inputScenes = data[0]
    targetScenes = data[1]


    # get patches from input images and targets
    inputList = []
    targetList = []
    for i in range(inputScenes.size(0)):
        helper = getPatches(inputScenes[i], patchSize, stride)
        inputList.append(helper)

        helper = getPatches(targetScenes[i], patchSize, stride)
        targetList.append(helper)
    print("start model predictions")
    # get predictions from input patches
    for i in range(len(inputList[0])):
        helperInpt = list(x[i] for x in inputList)
        targetInpt = list(x[i] for x in targetList)
        inputPatches = torch.stack(helperInpt, dim=0) # for vision Transformer; remove squeeze bla if not working
        targetPatches = torch.stack(targetInpt, dim=0)

        # predict with model

        modelInpt = inputPatches.squeeze().unsqueeze(dim = 0)

        prediction = model.forward(modelInpt, targetPatches, training=False).squeeze()


        # switch input with predictions; z = scene index, i = patch index
        for z in range(prediction.size(0)):
            inputList[z][i] = prediction[z, :, :]
            #inputList[z][i] = modelInpt[z, :, :]

    # set patches back to images
    scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride) for x in inputList)

    ## plot
    if plot:
        plotList = [targetScenes[d] for d in range(targetScenes.size(0))]
        plotList = plotList + scenePredictions
        plotList = [x.detach().cpu().numpy() for x in plotList]
        plotList = [np.transpose(x, (1,2,0)) for x in plotList]

        fig, axs = plt.subplots(2, 4)

        for i in range(8):
            ax = axs[i // 4, i % 4]
            ax.imshow(plotList[i])
            ax.axis('off')

        plt.show()
    if safe:
        print("start saving prediction scenes")

        path = pathOrigin + "/results"
        os.chdir(path)
        os.makedirs(modelName, exist_ok=True)
        os.chdir(path + "/" + modelName)

        os.makedirs("modelPredictions", exist_ok=True)
        os.chdir(os.getcwd() + "/modelPredictions")
        os.makedirs(glacierName, exist_ok=True)
        os.chdir(os.getcwd() + "/" + glacierName)
        os.makedirs(predictionName, exist_ok=True)
        os.chdir(os.getcwd() + "/" + predictionName)

        path = os.getcwd()
        for i in range(len(scenePredictions)):
            plt.clf()
            # model predictions
            os.chdir(path)
            os.makedirs("predictions", exist_ok=True)
            os.chdir(path + "/" + "predictions")
            plt.imshow(minmaxScaler(scenePredictions[i].cpu().detach().numpy()[0,:,:]), cmap='gray')

            # save on harddrive
            p = os.getcwd()+ "/" + str(i) + ".pdf"
            plt.savefig(p, dpi=1000)
            plt.clf()
            with open(str(i), "wb") as fp:  # Pickling
                pickle.dump(scenePredictions[i].cpu().detach().numpy(), fp)


            # target predictions
            os.chdir(path)
            os.makedirs("targets", exist_ok=True)
            os.chdir(path + "/" + "targets")
            plt.imshow(minmaxScaler(targetScenes[i].cpu().detach().numpy()[0, :, :]), cmap='gray')

            # save on harddrive
            p = os.getcwd() + "/" + str(i) + ".pdf"
            plt.savefig(p, dpi=1000)

            with open(str(i), "wb") as fp:  # Pickling
                pickle.dump(targetScenes[i].cpu().detach().numpy(), fp)

    print("prediction scenes saved")
    #return scenePredictions
    return


def saveCheckpoint(model, optimizer, filename):
    """
    saves current model and optimizer step

    model: nn.model
    optimizer: torch.optim.optimzer class
    filename: string
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(pathOrigin, filename))
    print("checkpoint saved")
    return


def loadCheckpoint(model, optimizer, path):
    """
    loads mode and optimzer for further training
    model: nn.model
    optimizer: torch.optim.optimzer class
    path: string
    return: list of optimizer and model

    """
    if optimizer != None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("checkpoint loaded")
        return [model, optimizer]
    elif optimizer == None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        return model


def tokenizerBatch(model, x, mode, device, flatten = torch.nn.Flatten(start_dim=1, end_dim=2)):
    """
    encoding and decoding function for transformer model

    model: nn.Module
    x: tensor
    mode: string
    device: string
    flatten: nn.Flatten for latent space input
    return: torch.tensor
        encoding/decoding for tarnsformer model
    """
    model.eval()
    if mode == "encoding":
        encoding = [model.encoder(x[i, :, :, :].to(device))[0] for i in range(x.size(0))]
        encoding = torch.stack(encoding).squeeze(dim = 2)

        return encoding

    if mode == "decoding":
        decoding = [model.decoder(x[i, :, :].to(device)) for i in range(x.size(0))]
        decoding = torch.stack(decoding)

        return decoding


def plotPatches(model, modelName, data, tokenizer, device, plot):
    """
    plots patches and targets and saves on harddrive

    model: nn.model object
    modelName: string
    data: list of tensor
    tokenizer: nn.object module
    plot: boolean
    """

    model.eval()
    x = data[0]
    # predictions

    if tokenizer:
        x = tokenizerBatch(tokenizer, data[0].to(device).float(), "encoding", device)
        #y = tokenizerBatch(tokenizer, data[1], "encoding", device)

        # forward + backward + optimize
        forward = model.forward(x,None, training=False)
        forward = tokenizerBatch(tokenizer, forward, "decoding", device)
        forward = torch.reshape(forward, (forward.size(0), forward.size(1), 50, 50))

    elif tokenizer == False:
        forward = model.forward(x, None, training=False)

    predictions = forward.unsqueeze(dim =0)
    targets = data[1].to(device).float()

    # put into list
    predList = []
    targetList = []
    for i in range(4):
        pred = predictions[:, i, :, :].detach().cpu().numpy().squeeze()
        predList.append(pred)

        targ = targets[:, i, :, :].detach().cpu().numpy().squeeze()
        targetList.append(targ)

    plotData = predList + targetList

    # check inference
    assert len(plotData) == 8

    # start plotting
    path = pathOrigin + "/predictions"
    os.chdir(path)
    os.makedirs(modelName, exist_ok= True)
    os.chdir(os.path.join(path, modelName))
    path = os.path.join(path, modelName)
    name = str(np.random.randint(50000))
    os.makedirs(name, exist_ok=True)
    os.chdir(path + "/" + name)

    path = os.getcwd()

    fig, axs = plt.subplots(ncols=int(len(plotData)/2), nrows=2)#, figsize=(20,30))
    axs = axs.flat
    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    for i in range(len(plotData)):
        # model predictions
        axs[i].imshow(minmaxScaler(plotData[i]), cmap='gray')
        axs[i].axis("off")
        #axs[i].axis('off')

    # save on harddrive
    plt.tight_layout()
    p = os.getcwd() + "/" + str(i) + ".pdf"
    plt.savefig(p, dpi=1000)

    with open(str(i), "wb") as fp:  # Pickling
        pickle.dump(plotData[i], fp)

    # Show the plot
    if plot:
        plt.show()
    return