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
import functions
import os
import pickle
import rasterio

# wrapper for acquiring and preprocessing data

def preprocessing(box, time, cloudCoverage, allowedMissings, gamma, year, plot = False):
    """
    acquire and preprocess the data

    box: tuple of float
        coordinate box from which images are taken
    time: string
        time range for extraction of data
    cloudCoverage: int
        percent of pixels covered with clouds in bands
    allowedMissings: float
        p(missingData)
    gamma: float
        parameter for image creation over time dimension, controls influence of past images
    year: string
        year of data extraction, downloads chunks of data as one year packages
    plot: boolean
        plot alignment (done correctly?)

    return: list of tuple of datetime and 4d ndarray tensor for model building
    """
    # get data
    res = functions.getData(bbox=box, bands=["red", "green", "blue", "swir16"], timeRange=time)
    d = functions.filterAndConvert(res, cloudCoverage, allowedMissings)

    print("start processing: ", len(d), " images")

    # apply kernel to raw bands, do two times as sometimes division by zero gives nans in image
    for x in range(2):
        for i in range(len(d)):
            for z in range(4):
                d[i][1][z, :, :] = functions.applyToImage(d[i][1][z, :, :])
    print("application of kernel done")

    # corners are completely missing in satelite images because of different paths of satelite,
    # cannot be resolved by kernel (division by zero), -> imputation of "mean image" pixels for corners
    for i in range(4):
        d = functions.imputeMeanValues(d, i)
    print("Imputation of mean values done")

    ## spatial alignment
    # alignment with SWIR band not possible -> values to small -> instead align NDSI image in [0,255] interval
    d = functions.NDSI(d)

    groundTruthInd = np.argmin(list(map(lambda x: x[2], d))) # take image with minimum cloud coverage as groundtruth
    # normal method
    # align bands spatially
    for i in range(4): # hard coded
        groundTruth = d[groundTruthInd][1][i, :, :]
        groundTruth = functions.minmaxConvertToRGB(groundTruth)

        # mixing parameter moving average over pixels, gamma conrols influence for past pixels in the creation of new pixels
        # -> get rid of corner artefacts
        img = functions.alignImages(functions.minmaxConvertToRGB(d[0][1][i, :, :]), groundTruth, False, 5000, 0.25)
        for x in range(len(d)):
            helper = functions.minmaxConvertToRGB(d[x][1][i, :, :], ret=True)
            curImage = helper[0]
            img = (gamma * img) + ((1 - gamma) * functions.alignImages(curImage, groundTruth, False, 5000, 0.25))
            d[x][1][i, :, :] = img

            # get masks for snow and no snow
            # get threshold of NDSI in RGB coordinates
            if i == 3:
                thresh = ((0.3 - helper[1]) / (helper[2] - helper[1])) * 255.999

                # mask images
                nosnow = np.ma.masked_where(d[x][1][i, :, :] >= thresh, d[x][1][i, :, :]).filled(0)
                snow = np.ma.masked_where(d[x][1][i, :, :] < thresh, d[x][1][i, :, :]).filled(0)
                d[x] = (d[x][0], np.dstack((d[x][1][0, :, :],d[x][1][1, :, :], d[x][1][2, :, :], d[x][1][3, :, :], nosnow, snow)))

            print("image: ", x, " of band: ", i, " done")

        print("preprocessing of full band: ", i, " done")

    """
    # align bands spatially
    numfeatures = np.arange(0,10000,100)
    kept = np.arange(0,1,0.01)
    res = []
    for i in range(4):  # hard coded
        groundTruth = d[groundTruthInd][1][i, :, :]
        groundTruth = functions.minmaxConvertToRGB(groundTruth)

        # mixing parameter moving average over pixels, gamma conrols influence fo past pixels in the creation of new pixels
        # -> get rid of corner artefacts
        img = functions.alignImages(functions.minmaxConvertToRGB(d[0][1][i, :, :]), groundTruth, False, 5000, 0.25)
        for x in range(len(d)):
            helper = functions.minmaxConvertToRGB(d[x][1][i, :, :], ret=True)
            curImage = helper[0]
            img = (gamma * img) + ((1 - gamma) * functions.alignImages(curImage, groundTruth, False, 5000, 0.25))
            d[x][1][i, :, :] = img

            # get masks for snow and no snow
            # get threshold of NDSI in RGB coordinates
            if i == 3:
                thresh = ((0.3 - helper[1]) / (helper[2] - helper[1])) * 255.999

                # mask images
                nosnow = np.ma.masked_where(img >= thresh, img).filled(0)
                snow = np.ma.masked_where(img < thresh, img).filled(0)
                d[x] = (d[x][0], np.dstack(
                    (d[x][1][0, :, :], d[x][1][1, :, :], d[x][1][2, :, :], d[x][1][3, :, :], nosnow, snow)))

            # print("image: ", x, " of band: ", i, " done")

        print("preprocessing of full band: ", i, " done")
    """






    # save on harddrive with pickling
    # create folder
    pathOrigin = os.getcwd()
    pathOrigin = os.path.join(pathOrigin, "images")
    os.makedirs(pathOrigin, exist_ok = True)
    os.chdir(pathOrigin)

    #plot as check
    if plot:
        for i in range(len(d)):
            plt.subplot(round(len(d)/2), 2, i + 1)
            plt.imshow(d[i][1][:, :, 0])
        plt.show()

        for i in range(len(d)):
            plt.subplot(round(len(d)/2), 2, i + 1)
            plt.imshow(d[i][1][:, :, 1])
        plt.show()

        for i in range(len(d)):
            plt.subplot(round(len(d)/2), 2, i + 1)
            plt.imshow(d[i][1][:, :, 2])
        plt.show()

        for i in range(len(d)):
            plt.subplot(round(len(d)/2), 2, i + 1)
            plt.imshow(d[i][1][:, :, 3])
        plt.show()

        for i in range(len(d)):
            plt.subplot(round(len(d)/2), 2, i + 1)
            plt.imshow(d[i][1][:, :, 4])
        plt.show()

        for i in range(len(d)):
            plt.subplot(round(len(d)/2), 2, i + 1)
            plt.imshow(d[i][1][:, :, 5])
        plt.show()

    # save data object
    with open(year, "wb") as fp:  # Pickling
        pickle.dump(d, fp)
    print("data saved!")
    return d

# start

path = os.getcwd()
years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
for b in range(len(years)):
    os.chdir(path)
    if b < 10:
        str = years[b] + "-01-01/" + years[b+1] + "-01-01"
        preprocessing((7.890586853027345, 46.382464893261165, 8.221206665039064, 46.57160668424229),
              str, 15, 0.3, 0.3, years[b], plot = True)
        print(years[b] + " done")
    if b == 10:
        print("finished!")
        break
        

"""
path = os.getcwd()
os.chdir(path)

preprocessing((7.890586853027345, 46.382464893261165, 8.221206665039064, 46.57160668424229),
              "2019-07-11/2019-10-11", 10, 0.3, 1, 0.4, "test")
"""
#print(openData()[2][1].shape)












