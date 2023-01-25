# packages
import coiled
import distributed
import dask
import pandas as pd
import pystac_client
import planetary_computer as pc
import ipyleaflet
import IPython.display as dsp
import geogif
from dateutil.parser import ParserError
import stackstac
import bottleneck
import dask
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from numpy import array
import cv2
import imutils
from torch import nn
from numpy import linalg as LA
from numpy import ma
import os
import pickle
from sklearn.feature_extraction import image
import torch.optim as optim
import torch
import sys
from torchvision import transforms
from PIL import Image


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
    stack = stackstac.stack(items, bounds_latlon=bbox)

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
    res = ((X - np.nanmin(X) )/ (np.nanmax(X) - np.nanmin(X))) *255.99
    res = res.astype("uint8")
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
    for i in range(len(Input)):
        tensor = Input[i][1][:, :, :]
        NDSI = np.divide(np.subtract(tensor[2, :, :], tensor[5, :, :]), np.add(tensor[2, :, :], tensor[5, :, :]))
        nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0)
        snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
        switchD = np.dstack((tensor[0, :, :],tensor[1, :, :],tensor[2, :, :],tensor[3, :, :],tensor[4, :, :],
                             tensor[5, :, :],tensor[6, :, :], NDSI, nosnow, snow))
        switchD = np.transpose(switchD, (2,0,1)) # switch dimensions back
        Input[i] = (Input[i][0], switchD)
    return Input

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

def saveCheckpoint(state, filename):
    torch.save(state, filename)
    print("model checkpoint saved")
    
def loadCheckpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("loading model complete")


def trainLoop(data, model, loadModel, modelName, lr, weightDecay, earlyStopping, epochs,
              validationSet, validationStep):
    """

    data: list of list of input data and dates and targets
    model: pytorch nn.class
    loadModel: boolean
    modelName: string
        .pth.tar model name on harddrive
    lr: float
    weightDecay: float
    earlyStopping: float
    criterionFunction: nn.lossfunction
    epochs: int
    validationSet: same as data
    validationStep: int
        timepoint when validation set is evaluated for early stopping regularization


    return: nn.class
        trained model
    """
    torch.autograd.set_detect_anomaly(True)
    runningLoss = 0
    stoppingCounter = 0
    lastLoss = 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    trainLosses = []
    validationLosses = []
    trainCounter = 0

    # load model
    if loadModel:
        loadCheckpoint(torch.load(modelName), model=model, optimizer=optimizer)
    model.train()
    
    for x in range(epochs):
        # get indices for epoch
        ix = np.arange(0, len(data), 1)
        ix = np.random.choice(ix, len(data), replace=False, p=None)

        for i in ix:
            # get data
            helper = data[i]
            y = helper[1][0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            forward = model.forward(helper)
            # predictions = forward[0].to(device='cuda')
            predictions = forward[0]
            loss = MSEpixelLoss(predictions, y) + forward[1]
            loss.backward()
            optimizer.step()
            trainCounter += 1

            # print loss
            runningLoss += loss.item()
            meanRunningLoss = runningLoss / trainCounter
            trainLosses.append(meanRunningLoss)

            if i % validationStep == 0 and i != 0:
                if validationSet != None:
                    # sample data
                    validationLoss = 0
                    for i in range(len(validationSet)):
                        helper = validationSet[i]
                        y = helper[1][0]

                        # forward + backward + optimize
                        forward = model.forward(helper)
                        # predictions = forward[0].to(device='cuda')
                        predictions = forward[0]

                        testLoss = MSEpixelLoss(predictions, y) + forward[1]
                        validationLoss += testLoss.item()
                        meanValidationLoss = validationLoss / len(validationSet)
                        validationLosses.append([meanValidationLoss, trainCounter]) # save trainCounter as well for comparison with interpolation
                        # of in between datapoints

                    print("current validation loss: ", meanValidationLoss)

                # early stopping
                if earlyStopping > 0:
                    if (meanValidationLoss - lastLoss) > earlyStopping:
                        stoppingCounter += 1

                    if stoppingCounter == 10:
                        print("model converged, early stopping")
                        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                        saveCheckpoint(checkpoint, modelName)
                        # save losses
                        dict = {"trainLoss": trainLosses, "validationLoss": [np.NaN for x in range(trainLosses)]}
                        trainResults = pd.DataFrame(dict)

                        # fill in validation losses with index
                        for i in range(len(validationLosses)):
                            trainResults.iloc[validationLosses[i][1], 1] = validationLosses[i][0]

                        # save dartaFrame to csv
                        trainResults.to_csv("resultsTraining.csv")
                        ############ find better solution than exiting the file
                        quit()
                        #######################################################

                    lastLoss = meanValidationLoss
            print("epoch: ", x, ", example: ", trainCounter, " current loss = ", meanRunningLoss)

    ## save model anyways in case it did not converge
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    saveCheckpoint(checkpoint, modelName)

    # save losses
    dict = {"trainLoss": trainLosses, "validationLoss" : [np.NaN for x in range(trainLosses)]}
    trainResults = pd.DataFrame(dict)

    # fill in validation losses with index
    for i in range(len(validationLosses)):
        trainResults.iloc[validationLosses[i][1], 1] = validationLosses[i][0]

    # save dartaFrame to csv
    trainResults.to_csv("resultsTrainingPatches.csv")

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


def combinePatches(patches, tensorShape, patchSize, stride=50):

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
        for z in range(img.shape[0]):
            while np.count_nonzero(np.isnan(img[z, :, :])) > 0:
                img[z, :, :] = applyToImage(img[z, :, :])
                #print("still missing ", np.count_nonzero(np.isnan(img[z, :, :])), " pixels")
            print("band: ", z, " of ", img.shape[0], "done")
    print("application of kernel done")

    # torch conversion
    img = torch.from_numpy(img)
    patches = getPatches(img, patchSize, stride=stride)
    out = torch.stack(patches, dim = 0)
    out = out.numpy()

    return out


def automatePatching(data, patchSize, maxPatches, roi, applyKernel):
    """
    creates image patches sampled from a region of interest

    data: list of tuple of datetime and np.array
        data extracted from API
    patchSize: tuple of int
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
        patches = createPatches(data[i][1][:, :, :], patchSize, maxPatches, roi, applyKernel=applyKernel)
        res.append((data[i][0], patches))

    return res

def getTrainTest(patches, window, inputBands, outputBands, numSequences):
    """

    patches: list of tuple of datetime and np.array
        data from API
    window: int
        length of sequences for model
    inputBands: list of int
    outputBands: list of int
    numSequences: int
        number of sequences sampled from scenes

    returns: list of list of input data, input date and target data, target date

    """
    dataList = []
    for i in range(numSequences):
        # create patches from random consecutive timepoints in the future
        ## take next n scenes
        #x = patches[i:i + window]
        #y = patches[i + window: i + (2 * window)]
        ## take random next scenes
        # sample 2*window consecutive indices
        seq = np.arange(0, len(patches))
        seq = np.random.choice(seq, 2*window, replace = False).tolist()
        seq.sort()
        seqX = seq[0:window]
        seqY = seq[window:]
        x = [patches[t] for t in seqX]
        y = [patches[t] for t in seqY]
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
        print("batch ", i, " done")
    # save data object on drive
    with open("trainData", "wb") as fp:  # Pickling
        pickle.dump(dataList, fp)
    print("data saved!")

    return dataList

# input 5, 3, 50, 50; targets: 5, 1, 50, 50
def fullSceneLoss(inputScenes, inputDates, targetScenes, targetDates, model, patchSize, stride, outputDimensions, test = False):
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
            prediction = model.forward(finalInpt, training = True)

            # switch input with predictions; z = scene index, i = patch index
            for z in range(prediction[0].size(0)):
                inputList[z][i] = prediction[0][z, :, :]

            # accumulate latent space losses
            latentSpaceLoss += prediction[1].item()

        # get final loss of predictions of the full scenes
        # set patches back to images
        scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride) for x in inputList)
        fullLoss = sum(list(map(lambda x,y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))
        fullLoss += latentSpaceLoss

        return fullLoss

    if test:
        scenePredictions = list(combinePatchesTransfer(x, outputDimensions, patchSize, stride) for x in inputList)
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


def fullSceneTrain(model, modelName, optimizer, data, epochs, patchSize, stride, outputDimensions):
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

    """

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
            y = helper[1][0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            loss = fullSceneLoss(helper[0][0], helper[0][1],
                                 helper[1][0], helper[1][1],
                                 model,
                                 patchSize,
                                 stride,
                                 outputDimensions,
                                 test = False)
            loss.backward()
            optimizer.step()
            trainCounter += 1

            # print loss
            runningLoss += loss.item()
            meanRunningLoss = runningLoss / trainCounter
            trainLosses.append(meanRunningLoss)
            print("epoch: ", x, ", example: ", trainCounter, " current loss = ", meanRunningLoss)

    ## save model anyways in case it did not converge
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    saveCheckpoint(checkpoint, modelName)

    # save losses
    dict = {"trainLoss": trainLosses}
    trainResults = pd.DataFrame(dict)

    # save dartaFrame to csv
    trainResults.to_csv("resultsTrainingScenes.csv")

## visualize network performance on full scenes
def inferenceScenes(model, data, patchSize, stride, outputDimensions, plot = False):
    """

    model: nn.class object
    data: same as above
    patchSize: int
    stride: int
    outputDimensions: tuple
    plot: boolean

    return: list of tensor
        predicted scenes
    """
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
    latentSpaceLoss = 0
    for i in range(len(inputList[0])):
        helperInpt = list(x[i] for x in inputList)
        targetInpt = list(x[i] for x in targetList)
        inputPatches = torch.stack(helperInpt, dim=0)
        targetPatches = torch.stack(targetInpt, dim=0)

        # put together for final input
        finalInpt = [[inputPatches, inputDates], [targetPatches, targetDates]]

        # predict with model
        prediction = model.forward(finalInpt, training=True)

        # switch input with predictions; z = scene index, i = patch index
        for z in range(prediction[0].size(0)):
            inputList[z][i] = prediction[0][z, :, :]

        # accumulate latent space losses
        latentSpaceLoss += prediction[1].item()

    # get final loss of predictions of the full scenes
    # set patches back to images
    scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride) for x in inputList)

    ## plot
    if plot:
        plotList = [data[0][0][d] for d in range(5)]
        plotList = plotList + scenePredictions
        plotList = [x.numpy() for x in plotList]

        # Create a figure with 2 rows and 5 columns
        fig, axs = plt.subplots(2, 5)

        # Assume that `images` is a list of 10 images
        for i in range(10):
            # Get the current axis
            ax = axs[i // 5, i % 5]
            # Plot the image on the current axis
            ax.imshow(plotList[i])
            # Remove the axis labels
            ax.axis('off')

        # Show the figure
        plt.show()

    return scenePredictions

















