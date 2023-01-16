# packages
import coiled
import distributed
import dask
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
#from torchvision import transforms
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
    stack = stackstac.stack(items, bounds_latlon=bbox, epsg = "EPSG:3267")

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

def createPatches(img, patchSize, maxPatches, roi, applyKernel = False):
    """
    creates image patches sampled from a region of interest

    img: np.array
        shape = (11, x,y)
    patchSize: tuple of int
        size of patches
    maxPatches: int
        number of patches extracted
    roi: list of int
        bounding box region of interest for importance sampling
    applyKernel: boolean
        if data still contains missings apply kernel to patches

    returns:  np.array
        shape = (n_patches, 11, pachsize[0], patchsize[1])
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

    # rearrange dims for scikit-learn
    img = np.transpose(img, (1,2,0))
    patches = image.extract_patches_2d(img, patchSize, max_patches=maxPatches, random_state=42)

    # get into pytorch fomat for conv2d
    # output of extract_patches_2d = (n_patches, patch_height, patch_width, n_channels)
    patches = np.transpose(patches, (0, 3, 1, 2))

    return patches


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


def trainLoop(data, model, loadModel, modelName, lr, weightDecay, earlyStopping, criterionFunction, maxIter, epochs,
              testSet):
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
    maxIter: int
    epochs: int

    return: nn.class
        trained model
    """
    torch.autograd.set_detect_anomaly(True)
    runningLoss = 0
    stoppingCounter = 0
    lastLoss = 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)

    for x in range(epochs):
        # load model
        if loadModel:
            loadCheckpoint(torch.load(modelName), model=model, optimizer=optimizer)
        model.train()
        for i in range(maxIter):
            # sample data
            idx = np.random.randint(0, len(data), size=1)
            helper = data[idx[0]]
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

            # print loss

            runningLoss += loss.item()
            if i % 100 == 0 and i != 0:
                if testSet != None:
                    # testLoss = [MSEpixelLoss(model.forward(x[0]).to("cuda"), x[1]).item() for x in testSet]
                    # testLoss = sum(testLoss)/len(testLoss)
                    # sample data
                    idx = np.random.randint(0, len(testSet), size=1)
                    helper = testSet[idx[0]]
                    y = helper[1][0]

                    # forward + backward + optimize
                    forward = model.forward(helper)
                    # predictions = forward[0].to(device='cuda')
                    predictions = forward[0]

                    testLoss = MSEpixelLoss(predictions, y) + forward[1]
                    print("current test loss: ", testLoss)
            print("epoch ", x, ", batch ", i, " current loss = ", runningLoss / (i + 1))

            # early stopping
            if earlyStopping > 0:
                if (lastLoss - loss.item()) < earlyStopping:
                    stoppingCounter += 1

                if stoppingCounter == 10:
                    print("model converged, early stopping")
                    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                    saveCheckpoint(checkpoint, modelName)
                    sys.exit()
            lastLoss = loss.item()

        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        saveCheckpoint(checkpoint, modelName)

    return model


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
            patch = tensor[:, y:y + patchSize, x:x + patchSize]
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
            tensor[:, y:y + patchSize, x:x + patchSize] += patch
            patchIndex += 1
    # Return the image tensor
    return tensor







