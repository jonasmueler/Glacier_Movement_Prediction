# packages
#import coiled
#import distributed
#import dask
import pandas as pd
import pystac_client
import planetary_computer as pc
#import ipyleaflet
#import IPython.display as dsp
#import geogif
#from dateutil.parser import ParserError
import stackstac
#import bottleneck
#import dask
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.image as mpimg
from numpy import array
import cv2
#import imutils
from torch import nn
#from numpy import linalg as LA
#from numpy import ma
import os
import pickle
#from sklearn.feature_extraction import image
import torch.optim as optim
import torch
# memory overflow bug fix
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
#
#import sys
#from torchvision import transforms
from PIL import Image
import wandb
from torch.autograd import Variable


## global variables for project
### change here to run on cluster ####
pathOrigin = "/mnt/qb/work/ludwig/lqb875"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"



def getData(bbox, bands, timeRange, cloudCoverage, allowedMissings):
    """
    gets data in numpy format

    bbox: list of float
        rectangle to be printed
    bands: list of string
        ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'ST_QA',
       'lwir11', 'ST_DRAD', 'ST_EMIS', 'ST_EMSD', 'ST_TRAD', 'ST_URAD',
       'QA_PIXEL', 'ST_ATRAN', 'ST_CDIST', 'QA_RADSAT', 'SR_QA_AEROSOL']
    timeRange: string
        e.g. "2020-12-01/2020-12-31"
    cloudCoverage: int
        amount of clouds allowed in %
    allowedMissings: float
        amount of pixels nan

    returns: list of tuple of datetime array and 4d numpy array and cloudCoverage array
        [time, bands, x, y]

    """

    catalog = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')

    # query
    search = catalog.search(
        collections=['landsat-8-c2-l2'],
        max_items= None,
        bbox=bbox,
        datetime=timeRange
    )

    items = pc.sign(search)
    print("found ", len(items), " scenes")

    # stack
    stack = stackstac.stack(items, bounds_latlon=bbox, epsg = "EPSG:32625")

    # use common_name for bands
    stack = stack.assign_coords(band=stack.common_name.fillna(stack.band).rename("band"))
    output = stack.sel(band=bands)

    # put into dataStructure
    t = output.shape[0]
    cloud = np.array(output["eo:cloud_cover"] <= cloudCoverage)
    cloud = [cloud, np.array(output["eo:cloud_cover"])]
    time = np.array(output["time"])

    dataList = []
    for i in range(t):
        if cloud[0][i] == True:  # check for clouds
            if np.count_nonzero(np.isnan(output[i, 1, :, :])) >= round(
                    (output.shape[2] * output.shape[3]) * allowedMissings):  # check for many nans
                pass
            elif np.count_nonzero(np.isnan(output[i, 1, :, :])) <= round(
                    (output.shape[2] * output.shape[3]) * allowedMissings):
                data = array([np.array(output[i, 0, :, :]),
                              np.array(output[i, 1, :, :]),
                              np.array(output[i, 2, :, :]),
                              np.array(output[i, 3, :, :]),
                              np.array(output[i, 4, :, :]),
                              np.array(output[i, 5, :, :]),
                              np.array(output[i, 6, :, :])
                              ])
                cloudCov = cloud[1][i]
                data = (time[i], data, cloudCov)
                dataList.append(data)

    return dataList

# visualize RGB images
# convert to RGB scale

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
    res = ((X - np.nanmin(X) )/ (np.nanmax(X) - np.nanmin(X))) #*255.99
    #res = res.astype("uint8")
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


# calculate NDSI values; bands not correct, check !! adapt if all eleven bands are used
def NDSI(Input, threshold):
    """
    creates three new images: NDSI, snow-mask, no-snow-mask

    Input: list of tuple of datetime and 3d ndarray
    threshold: float
        threshold for NDSI masks ~ 0.3-0.6 usually

    returns: list of tuple of datetime and 3d ndarray
        switch swir Band with calculated NDSI values
    """
    """
    for i in range(len(Input)):
        #tensor = Input[i][1][:, :, :]
        tensor = Input
        NDSI = np.divide(np.subtract(tensor[2, :, :], tensor[5, :, :]), np.add(tensor[2, :, :], tensor[5, :, :]))
        nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0)
        snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
        switchD = np.dstack((tensor[0, :, :],tensor[1, :, :],tensor[2, :, :],tensor[3, :, :],tensor[4, :, :],
                             tensor[5, :, :],tensor[6, :, :], NDSI, nosnow, snow))
        switchD = np.transpose(switchD, (2,0,1)) # switch dimensions back
        Input[i] = (Input[i][0], switchD)
    """
    tensor = Input
    NDSI = np.divide(np.subtract(tensor[2, :, :], tensor[5, :, :]), np.add(tensor[2, :, :], tensor[5, :, :]))
    nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0)
    snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
    switchD = np.dstack((tensor[0, :, :], tensor[1, :, :], tensor[2, :, :], tensor[3, :, :], tensor[4, :, :],
                         tensor[5, :, :], tensor[6, :, :], NDSI, nosnow, snow))
    switchD = np.transpose(switchD, (2, 0, 1))  # switch dimensions back

    return switchD

# blur imputed pixels to cover edge of imputation
def gaussianBlurring(Input, kSize, band):
    """
    Input: list of tuple of datetime and 3d ndarray

    returns: list of tuple of datetime and 3d ndarray
        with applied filter on band 1 and 3
    """
    for i in range(len(Input)):
        Input[i][1][band, :, :] = cv2.GaussianBlur(Input[i][1][band, :, :], (kSize, kSize), 0)
    return Input


### image alignment with ORB features and RANSAC algorithm (see paper), same parameters used
def alignImages(image, template, RGB, maxFeatures, keepPercent):
    """
    image: 2d or 3d nd array
        input image to be aligned
    template: 2d or 3d nd array
        template for alignment
    RGB: boolean
        is image in RGB format 3d ndarray?
    maxFeatures: int
        max. amount of features used for alignment
    keepPercent: float
        amount of features kept for aligning



    returns: ndarray
        alignend image

    """

    # convert both the input image and template to grayscale
    if RGB:
        imageGray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
        templateGray =  cv2.cvtColor(template.astype('uint8'), cv2.COLOR_BGR2GRAY)
    if RGB == False:
        imageGray = image
        imageGray = imageGray.astype('uint8')

        templateGray = template
        templateGray = templateGray.astype('uint8')

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)


    # match the features
    #method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    #matcher = cv2.DescriptorMatcher_create(method)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descsA, descsB) #, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]


    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    #print(ptsA)
    #print(ptsB)
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]

    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned

def openData(name):
    """
    opens pickled data object
    
    name : string
    	named of saved data object
    	
    returns : list of tuple of datetime and np array 

    """
    with open(name, "rb") as fp:   # Unpickling
        data = pickle.load(fp)
    return data

def loadData(path, years):
    """

    path: string
        path to data pickle objects
    years: list of string
        years to be loaded
    returns: list of tuple of datetime and np.array
        date and image in list of tuple
    """

    os.chdir(path)
    Path = os.getcwd()
    os.chdir(Path)
    print("Begin loading data")

    # read
    fullData = []
    for i in range(len(years)):
        helper = openData(years[i])
        fullData.append(helper)
    print("data loading finished")

    d = [item for sublist in fullData for item in sublist]

    return d

## create time series of 5 images
def convertDatetoVector(date):
    """

    date: dateTime object
        time of picture taken
    returns: np.array
        vector of date
    """
    date = str(date)
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    res = torch.tensor([day, month, year], dtype = torch.float)

    return res

def MSEpixelLoss(predictions, y):
    """

    predictions: tensor
        dims = (seqLen, x,y)
    y: list of (x,y)

    return: float
        pixel loss for all images
    """

    #y = torch.stack(y, 0)
    y = y.to(torch.float32)

    loss = torch.nn.MSELoss()(predictions, y)

    return loss

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
    torch.save(checkpoint, filename)
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
    checkpoint = torch.load(path)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    print("model complete")
    return [model, optimizer]


def trainLoop(data, model, loadModel, modelName, lr, weightDecay, earlyStopping, epochs,
              validationSet, validationStep, WandB, device, pathOrigin = pathOrigin):
    """

    data: list of list of input data and dates and targets
    model: pytorch nn.class
    loadModel: boolean
    modelName: string
        .pth.tar model name on harddrive with path
    lr: float
    weightDecay: float
    earlyStopping: float
    criterionFunction: nn.lossfunction
    epochs: int
    validationSet: same as data
    validationStep: int
        timepoint when validation set is evaluated for early stopping regularization
    WandB: boolean
        use weights and biases tool to monitor losses dynmaically
    device: string
        device on which the data should be stored


    return: nn.class
        trained model
    """
    torch.autograd.set_detect_anomaly(True)
    runningLoss = 0
    runningLossLatentSpace = np.zeros(len(data) * epochs)
    meanRunningLossLatentSpace = 0
    runningLossReconstruction = np.zeros(len(data) * epochs)
    meanRunningLossReconstruction = 0
    stoppingCounter = 0
    lastLoss = 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    trainLosses = np.zeros(len(data) * epochs)
    validationLosses = np.zeros((len(data) * epochs, 2))
    validationLoss = 0
    trainCounter = 0
    trainCounterValidation = 0
    meanValidationLoss = 0


    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project= modelName,

            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": modelName,
                "dataset": "Helheim, Aletsch, jakobshavn",
                "epochs": epochs,
            }
        )

    # load model
    if loadModel:
        # get into folder
        os.chdir(pathOrigin + "/results/" + modelName)
        lastState = loadCheckpoint(model, optimizer, pathOrigin + "/results/" + modelName + "/" + modelName)
        model = lastState[0]
        optimizer = lastState[1]
    model.train()
    
    for x in range(epochs):
        # get indices for epoch
        ix = np.arange(0, len(data), 1)
        ix = np.random.choice(ix, len(data), replace=False, p=None)

        for i in ix:
            # get data
            helper = data[i]

            # move to cuda
            helper = moveToCuda(helper, device)

            #define target
            y = helper[1][0]

            # zero the parameter gradients
            optimizer.zero_grad()
            model.zero_grad()

            # forward + backward + optimize
            forward = model.forward(helper, training = True)
            predictions = forward[0]
            loss = MSEpixelLoss(predictions, y) + forward[1] + forward[2] # output loss, latent space loss, recopnstruction loss
            loss.backward()
            optimizer.step()
            trainCounter += 1

            # print loss
            meanRunningLossLatentSpace += forward[1].item()
            meanRunningLossLatentSpace = meanRunningLossLatentSpace/trainCounter
            runningLossLatentSpace[trainCounter - 1] = meanRunningLossLatentSpace
            meanRunningLossReconstruction += forward[2].item()
            meanRunningLossReconstruction = meanRunningLossReconstruction/trainCounter
            runningLossReconstruction[trainCounter - 1] = meanRunningLossReconstruction
            runningLoss += loss.item()
            meanRunningLoss = runningLoss / trainCounter
            trainLosses[trainCounter - 1] = meanRunningLoss

            ## log to wandb
            if WandB:
                wandb.log({"train loss": meanRunningLoss,
                           "latentSpaceLoss": meanRunningLossLatentSpace,
                           "reconstructionLoss": meanRunningLossReconstruction,
                           "validationLoss": meanValidationLoss})


            if i % validationStep == 0 and i != 0:
                if validationSet != None:
                    # sample validation set datum
                    ind = np.random.randint(0, len(validationSet))
                    helper = validationSet[ind]

                    # move to cuda
                    helper = moveToCuda(helper, device)

                    y = helper[1][0]

                    # forward
                    forward = model.forward(helper, training=True)
                    # predictions = forward[0].to(device='cuda')
                    predictions = forward[0]
                    trainCounterValidation += 1
                    testLoss = MSEpixelLoss(predictions, y) + forward[1] + forward[2]
                    validationLoss += testLoss.item()
                    meanValidationLoss = validationLoss / trainCounterValidation
                    validationLosses[trainCounter - 1] = np.array([meanValidationLoss, trainCounter])  # save trainCounter as well for comparison with interpolation
                    # of in between datapoints

                    # save memory
                    #del forward, helper

                    print("current validation loss: ", meanValidationLoss)

                # early stopping
            if earlyStopping > 0:
                if lastLoss < meanValidationLoss:
                    stoppingCounter += 1

                if stoppingCounter == 1000:
                    print("model converged, early stopping")

                    # navigate/create order structure
                    path = pathOrigin + "/results"
                    os.chdir(path)
                    os.makedirs(modelName, exist_ok=True)
                    os.chdir(path + "/" + modelName)
                    saveCheckpoint(model, optimizer, modelName)

                    # save losses
                    dict = {"trainLoss": trainLosses, "validationLoss": [np.NaN for x in range(len(trainLosses))]}
                    trainResults = pd.DataFrame(dict)

                    # fill in validation losses with index
                    for i in range(len(validationLosses)):
                        trainResults.iloc[int(validationLosses[i, 1].item()), 1] = validationLosses[i, 0].item()

                    # save dartaFrame to csv
                    trainResults.to_csv("resultsTraining.csv")
                    return

            lastLoss = meanValidationLoss

            print("epoch: ", x, ", example: ", trainCounter, " current loss = ", meanRunningLoss)
            #print("epoch: ", x, ", example: ", trainCounter, " current loss = ", loss.item())

            # save memory
            del loss, forward, helper, y

    path = pathOrigin + "/results"
    os.chdir(path)
    os.makedirs(modelName, exist_ok = True)
    os.chdir(path + "/" + modelName)

    ## save model anyways in case it did not converge
    saveCheckpoint(model, optimizer, modelName)

    # save losses
    dict = {"trainLoss": trainLosses,
            "validationLoss" : [np.NaN for x in range(len(trainLosses))],
            "latentSpaceLoss": runningLossLatentSpace,
            "reconstructionLoss": runningLossReconstruction}
    trainResults = pd.DataFrame(dict)

    # fill in validation losses with index
    for i in range(len(validationLosses)):
        trainResults.iloc[int(validationLosses[i, 1].item()), 1] = validationLosses[i, 0].item()

    # save dartaFrame to csv
    trainResults.to_csv("resultsTrainingPatches.csv")

    print("results saved!")
    return

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

def createPatches(img, patchSize, stride, roi, applyKernel = False):
    """
    creates image patches sampled from a region of interest

    img: np.array
    patchSize: int
        size of patches
    stride: int
    roi: list of int
        bounding box region of interest for importance sampling
    applyKernel: boolean
        if data still contains missings apply kernel to patches

    returns:  torch.tensor
        shape = (n_patches, bands, patchSize, patchSize)
    """

    img = img[:, roi[0]:roi[1], roi[2]:roi[3]]  # roi x,y coordinates over all bands
    # clean missings
    if applyKernel:
        # apply kernel to raw bands, do many times as sometimes division by zero gives nans in image
        for z in [2,5]: # only use NDSI relevant bands
            while np.count_nonzero(np.isnan(img[z, :, :])) > 0:
                img[z, :, :] = applyToImage(img[z, :, :])
                print("still missing ", np.count_nonzero(np.isnan(img[z, :, :])), " pixels")
            print("band: ", z, " of ", img.shape[0], "done")
    print("application of kernel done")

    # apply NDSI here
    # add NDSI and snow masks
    img = NDSI(img, 0.3)  # hardcoded snow value threshold

    # torch conversion, put into ndarray
    img = torch.from_numpy(img)
    patches = getPatches(img, patchSize, stride=stride)
    out = torch.stack(patches, dim = 0)
    out = out.numpy()

    return out


def automatePatching(data, patchSize, stride, roi, applyKernel):
    """
    creates image patches sampled from a region of interest

    data: list of tuple of datetime and np.array
        data extracted from API
    patchSize: int
        size of patches
    maxPatches: int
        number of patches extracted
    roi: list of int
        bounding box region of interest for importance sampling
    applyKernel: boolean
        if data still contains missings apply kernel to patches

    returns:  list of tuple of datetime and np.array
        switch np array in tuple with np array with on more dimension -> patches
    """

    res = []
    for i in range(len(data)):
        print("processing image: ", i)
        patches = createPatches(data[i][1][:, :, :], patchSize, stride, roi, applyKernel=applyKernel)
        res.append((data[i][0], patches))

    return res

def getTrainTest(patches, window, inputBands, outputBands):
    """
    takes 5 relative time deltas between scenes and outputs patch sequences with their corresponding date vectors

    patches: list of list of tensor and tensor and list of tensor and tensor
        data createPatches.py
    window: int
        length of sequences for model
    inputBands: list of int
    outputBands: list of int

    returns: list of list of input data, input date and target data, target date

    """
    dataList = []
    deltas = np.arange(1,6,1) # [1:5]
    counter = 0
    for delta in deltas:
        patchList = patches[::delta]
        for i in range((len(patchList) - 2*window) // 1 + 1): # formula from pytorch cnn classes
            # create patches from random consecutive timepoints in the future
            ## take next n scenes
            x = patchList[i:i + window]
            y = patchList[i + window: i + (2 * window)]
            ## take random next scenes
            # sample 2*window consecutive indices
            #seq = np.arange(0, len(patches))
            #seq = np.random.choice(seq, 2*window, replace = False).tolist()
            #seq.sort()
            #seqX = seq[0:window]
            #seqY = seq[window:]
            #x = [patches[t] for t in seqX]
            #y = [patches[t] for t in seqY]
            for z in range(x[0][1].shape[0]):
                xDates = [convertDatetoVector(x[t][0]) for t in range(len(x))]
                xDates = torch.stack(xDates, dim=0)
                yDates = [convertDatetoVector(y[t][0]) for t in range(len(y))]
                yDates = torch.stack(yDates, dim=0)
                xHelper = list(map(lambda x: torch.from_numpy(x[1][z, inputBands, :, :]), x))
                xHelper = torch.stack(xHelper, dim = 0)
                yHelper = list(map(lambda x: torch.from_numpy(x[1][z, outputBands, :, :]), y))
                yHelper = torch.stack(yHelper, dim=0)

                # sanity checks
                assert len(xDates) == len(yDates) == len(xHelper) == len(yHelper)

                # save
                dataList.append([[xHelper, xDates], [yHelper, yDates]])

        print("delta ", counter, " done")
        counter += 1

    # save data object on drive
    with open("trainData", "wb") as fp:  # Pickling
        pickle.dump(dataList, fp)
    print("data saved!")

    return dataList

# input 5, 3, 50, 50; targets: 5, 1, 50, 50
def fullSceneLoss(inputScenes, inputDates, targetScenes, targetDates,
                  model, patchSize, stride, outputDimensions, device = "cpu", training = True,
                  test = False):
    """
    train model on loss of full scenes and backpropagate full scene error in order to get smooth boarders in the final scene predictions

    inputScenes: tensor
        scenes
    inputDates: tensor
        dates
    targetScenes: tensor
        target scenes
    targetDates: tensor
        target dates
    model: torch.model object
    patchSize: int
    stride: int
        used stride for patching
    outputDimensions: tuple
        dimensions of output scenes
    device: string
        on which device is tensor calculated
    training: boolean
        inference?
    test: boolean
        test pipeline without model predictions


    returns: int
        loss on full five scenes and all associated patches

    """

    # get patches from input images and targets
    inputList = []
    targetList = []
    for i in range(inputScenes.size(0)):
        helper = getPatches(inputScenes[i], patchSize, stride)
        inputList.append(helper)

        helper = getPatches(targetScenes[i], patchSize, stride)
        targetList.append(helper)

    # get predictions from input patches
    if test == False:
        latentSpaceLoss = 0
        for i in range(len(inputList[0])):
            helperInpt = list(x[i] for x in inputList)
            targetInpt = list(x[i] for x in targetList)
            inputPatches = torch.stack(helperInpt, dim = 0)
            targetPatches = torch.stack(targetInpt, dim=0)


            # put together for final input
            finalInpt = [[inputPatches, inputDates], [targetPatches, targetDates]]

            # predict with model
            prediction = model.forward(finalInpt, training = training)

            # switch input with predictions; z = scene index, i = patch index
            for z in range(prediction[0].size(0)):
                inputList[z][i] = prediction[0][z, :, :]

            # accumulate latent space losses
            latentSpaceLoss += prediction[1].item()

        # get final loss of predictions of the full scenes
        # set patches back to images
        scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride, device = device) for x in inputList)
        fullLoss = sum(list(map(lambda x,y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))
        fullLoss += latentSpaceLoss


        # save memory
        #del prediction
        #del scenePredictions

        return fullLoss

    if test:
        scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride) for x in inputList)
        fullLoss = sum(list(map(lambda x, y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))
        return fullLoss

        latentSpaceLoss = 0
        for i in range(len(inputList[0])):
            helperInpt = list(x[i] for x in inputList)
            targetInpt = list(x[i] for x in targetList)
            inputPatches = torch.stack(helperInpt, dim=0)
            targetPatches = torch.stack(targetInpt, dim=0)

            # use targets in order to test pipeline without model prediction, takes 5 images extracts pathces and puts images back again
            # without model predictions loss should be 0 if pipeline works
            for z in range(inputPatches.size(0)):
                inputList[z][i] = targetPatches[z, :, :]


        # get final loss of predictions of the full scenes
        # set patches back to images
        scenePredictions = list(combinePatchesTransfer(x, outputDimensions, patchSize, stride) for x in inputList)
        fullLoss = sum(list(map(lambda x, y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))
        return fullLoss


def fullSceneTrain(model, modelName, optimizer, data, epochs, patchSize, stride, outputDimensions, device,
                   WandB, pathOrigin = pathOrigin):
    """

    train model on full scenes

    model: torch nn.model
    modelName: string
    optimizer: torch optim object
    data: list of list of tensor, tensor and tensor, tensor
        five scenes input and dates, five scenes targets and dates
    epochs: int
    patchSize: int
    stride: int
    outputDimensions: tuple
    device: string
        machine to be used
    pathOrigin: str
        path for data safing

    """
    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project=modelName,

            # track hyperparameters and run metadata
            config={
                "architecture": modelName,
                "dataset": "Helheim, Aletsch, jakobshavn",
                "epochs": epochs,
            }
        )


    trainCounter = 0
    runningLoss = 0
    trainLosses = []
    for x in range(epochs):
        # get indices for epoch
        ix = np.arange(0, len(data), 1)
        ix = np.random.choice(ix, len(data), replace=False, p=None)

        for i in ix:
            # get data
            helper = data[i]

            # move to cuda
            helper = moveToCuda(helper, device)

            y = helper[1][0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # use only target prediction for loss, latent space should already be learned, just edges of patches are smoothed
            loss = fullSceneLoss(helper[0][0], helper[0][1],
                                 helper[1][0], helper[1][1],
                                 model,
                                 patchSize,
                                 stride,
                                 outputDimensions,
                                 device=device,
                                 training = True,
                                 test = False)
            loss = torch.divide(loss, 144) # normalize loss
            loss.backward()
            optimizer.step()
            trainCounter += 1

            # print loss
            runningLoss += loss.item()
            meanRunningLoss = runningLoss / trainCounter
            trainLosses.append(meanRunningLoss)

            ## log to wandb
            if WandB:
                wandb.log({"train loss": meanRunningLoss})

            # save memory
            #del loss

            print("epoch: ", x, ", example: ", trainCounter, " current loss = ", meanRunningLoss)

    path = pathOrigin + "/results"
    os.chdir(path)
    os.makedirs(modelName, exist_ok=True)
    os.chdir(path + "/" + modelName)

    ## save model
    saveCheckpoint(model, modelName)

    # save losses
    dict = {"trainLoss": trainLosses}
    trainResults = pd.DataFrame(dict)

    # save dataFrame to csv
    trainResults.to_csv("resultsTrainingScenes.csv")

    return

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
    # move to cuda
    data = moveToCuda(data, device)

    inputScenes = data[0][0]
    targetScenes = data[1][0]
    inputDates = data[0][1]
    targetDates = data[1][1]


    # get patches from input images and targets
    inputList = []
    targetList = []
    for i in range(inputScenes.size(0)):
        helper = getPatches(inputScenes[i], patchSize, stride)
        inputList.append(helper)

        helper = getPatches(targetScenes[i], patchSize, stride)
        targetList.append(helper)

    # get predictions from input patches
    for i in range(len(inputList[0])):
        helperInpt = list(x[i] for x in inputList)
        targetInpt = list(x[i] for x in targetList)
        inputPatches = torch.stack(helperInpt, dim=0)
        targetPatches = torch.stack(targetInpt, dim=0)

        # put together for final input
        finalInpt = [[inputPatches, inputDates], [targetPatches, targetDates]]

        # predict with model
        prediction = model.forward(finalInpt, training=False)

        # switch input with predictions; z = scene index, i = patch index
        for z in range(prediction.size(0)):
            inputList[z][i] = prediction[z, :, :]

    # get final loss of predictions of the full scenes
    # set patches back to images
    scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride) for x in inputList)

    ## plot
    if plot:
        plotList = [data[0][0][d] for d in range(5)]
        plotList = plotList + scenePredictions
        plotList = [x.detach().numpy() for x in plotList]
        plotList = [np.transpose(x, (1,2,0)) for x in plotList]

        fig, axs = plt.subplots(2, 5)

        for i in range(10):
            ax = axs[i // 5, i % 5]
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
            # model predictions
            os.chdir(path)
            os.makedirs("predictions", exist_ok=True)
            os.chdir(path + "/" + "predictions")
            plt.imshow(minmaxScaler(scenePredictions[i].cpu().detach().numpy()[0,:,:]), cmap='gray')

            # save on harddrive
            p = os.getcwd()+ "/" + str(i) + ".pdf"
            plt.savefig(p, dpi=1000)

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


def moveToCuda(y, device):
    """
    transfers datum to gpu/cpu

    y: list of list of tensor and tensor and list of tensor and tensor
        input datum
    return: list of list of tensor and tensor and list of tensor and tensor
        transferred to cuda gpu
    """

    y[0][0] = y[0][0].to(device).to(torch.float32).requires_grad_()
    y[0][1] = y[0][1].to(device).to(torch.float32).requires_grad_()
    y[1][0] = y[1][0].to(device).to(torch.float32).requires_grad_()
    y[1][1] = y[1][1].to(device).to(torch.float32).requires_grad_()

    return y


def loadFullSceneData(path, names, window, inputBands, outputBands, ROI, applyKernel):
    """
    creates dataset of full scenes in order to train model on full scene loss

    path: string
        path to dataset
    names: list of string
        names of files to load
    inputBands: int
        number of the input to be used
    outputBands: int
        number of bands to be used in the output

    return: list of tensor and tensor, and tensor and tensor
        datum = list of scenes, dates and targets and dates
    """
    d = loadData(path, names)


    # crop to ROIs
    for i in range(len(d)):
        d[i] = (d[i][0], d[i][1][:, ROI[0]:ROI[1], ROI[2]:ROI[3]])

    # kernel for nans
    if applyKernel:
        for i in range(len(d)):
            img = d[i][1]
            for z in [2,5]:
                while np.count_nonzero(np.isnan(img[z, :, :])) > 0:
                    img[z, :, :] = applyToImage(img[z, :, :])
                    print("still missing ", np.count_nonzero(np.isnan(img[z, :, :])), " pixels")
                print("band: ", z, " of ", img.shape[0], "done")
            d[i] = (d[i][0], NDSI(img, 0.3)) # add NDSI here
    print("application of kernel done")

    dataList = []
    deltas = np.arange(1, 6, 1)  # [1:5]
    counter = 0
    for delta in deltas:
        sceneList = d[::delta]
        for i in range((len(sceneList) - 2 * window) // 1 + 1):  # formula from pytorch cnn classes
            # create patches from random consecutive timepoints in the future
            ## take next n scenes
            x = sceneList[i:i + window]
            y = sceneList[i + window: i + (2 * window)]

            # dates
            xDates = [convertDatetoVector(x[t][0]) for t in range(len(x))]
            xDates = torch.stack(xDates, dim=0)
            yDates = [convertDatetoVector(y[t][0]) for t in range(len(y))]
            yDates = torch.stack(yDates, dim=0)

            # ROI in scenes
            xHelper = list(map(lambda x: torch.from_numpy(x[1][inputBands, :, :]), x))
            xHelper = torch.stack(xHelper, dim=0)
            yHelper = list(map(lambda x: torch.from_numpy(x[1][outputBands, :, :]), y))
            yHelper = torch.stack(yHelper, dim=0).unsqueeze(1)

            # sanity checks
            assert len(xDates) == len(yDates) == len(xHelper) == len(yHelper)

            # save
            dataList.append([[xHelper, xDates], [yHelper, yDates]])

        print("delta ", counter, " done")
        counter += 1

    # save data object on drive
    with open("trainDataFullScenes", "wb") as fp:  # Pickling
        pickle.dump(dataList, fp)
    print("data saved!")

    return dataList

## test
"""
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn"
dat = loadFullSceneData(path, ["2013", "2014"], 5, [7,8,9], 9, [50, 650, 100, 700], True)

# ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
os.chdir("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets")
# save data object on drive
with open("test", "wb") as fp:  # Pickling
    pickle.dump(dat, fp)
print("data saved!")
"""


def plotPatches(model, data, path, plot):
    """
    plots patches and targets and saves on harddrive

    model: nn.model object
    data: list of list of tensor and tensor, and list of tensor and tensor
    path: str
    plot: boolean
    """


    model.eval()

    # predictions
    forward = model.forward(data, training=False)
    predictions = forward

    # put into list
    predList = []
    targetList = []
    for i in range(5):

        pred = predictions[i].detach().cpu().numpy().squeeze()
        predList.append(pred)

        targ = data[1][0][i].detach().cpu().numpy().squeeze()
        targetList.append(targ)

    plotData = predList + targetList


    # integrate dates here

    # check inference
    assert len(plotData) == 10

    # start plotting
    path = pathOrigin + "/predictions"
    os.chdir(path)
    name = str(np.random.randint(50000))
    os.makedirs(name, exist_ok=True)
    os.chdir(path + "/" + name)

    path = os.getcwd()
    for i in range(len(plotData)):
        # model predictions
        plt.imshow(minmaxScaler(plotData[i]), cmap='gray')

        # save on harddrive
        p = os.getcwd() + "/" + str(i) + ".pdf"
        plt.savefig(p, dpi=1000)

        with open(str(i), "wb") as fp:  # Pickling
            pickle.dump(plotData[i], fp)

    # Show the plot
    if plot:
        plt.show()

    return


















