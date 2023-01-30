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

# wrapper for acquiring data

def API(box, time, cloudCoverage, allowedMissings, year, glacierName, plot = False):
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
    year: string
        year of data extraction, downloads chunks of data as one year packages
    glacierName: string
        Name of the glacier for folder structure
    plot: boolean
        plot alignment (done correctly?)

    return: list of tuple of datetime and 4d ndarray tensor for model building
    """
    # get data
    d = functions.getData(bbox=box, bands=['coastal', 'red', 'green', 'blue', 'nir08', 'swir16', 'swir22'], timeRange=time, cloudCoverage= cloudCoverage, allowedMissings=allowedMissings)
    #d = functions.filterAndConvert(res, cloudCoverage, allowedMissings)

    print("start processing: ", len(d), " images")
    """
    # apply kernel to raw bands, do many times as sometimes division by zero gives nans in image
    for i in range(len(d)):
        for z in range(7):
            while np.count_nonzero(np.isnan(d[i][1][z, :, :])) > 0:
                d[i][1][z, :, :] = functions.applyToImage(d[i][1][z, :, :])
                print("still missing ", np.count_nonzero(np.isnan(d[i][1][z, :, :])), " pixels")
        print("image: ", i ," of ", len(d), "done" )
    print("application of kernel done")

    #### sanity check images contain no missings at all ########
    # corners are completely missing in satelite images because of different paths of satelite,
    # cannot be resolved by kernel (division by zero), -> imputation of "mean image" pixels for corners
    for i in range(7):
        d = functions.imputeMeanValues(d, i)
    print("Imputation of mean values done")
    """
    # add NDSI and snow masks
    d = functions.NDSI(d, 0.3)
    # save on hard drive with pickling
    # create folder

    pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets"
    pathOrigin = pathOrigin + "/" + glacierName
    os.makedirs(pathOrigin, exist_ok = True)
    os.chdir(pathOrigin)

    #plot as check in RGB
    if plot: # sanity check
        rgb = functions.createImage(d[1][1][1:4, :, :], 0.40)
        plt.imshow(rgb)
        plt.show()
        for i in range(len(d)): #rgb
            rgb = functions.createImage(d[i][1][1:4,:,:], 0.40)
            plt.subplot(round(len(d)/2)+1, 2, i + 1)
            plt.imshow(rgb)
        plt.show()

    # save data object
    with open(year, "wb") as fp:  # Pickling
        pickle.dump(d, fp)
    print("data saved!")
    return d


# testing
"""
#box, time, cloudCoverage, allowedMissings, gamma, year, plot = False)
path = os.getcwd()
os.chdir(path)

preprocessing((-50.80013494364671, 69.07690195189494, -50.13408880106858, 69.27219472296017),
              "2018-06-10/2018-07-01", 20, 0.3, "test", plot = True)

"""



# start
path = os.getcwd()
years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]

for b in range(len(years)):
    os.chdir(path)
    if b < 10:
        str = years[b] + "-01-01/" + years[b+1] + "-01-01"
        API((-49.707011136141695, 69.0891590033335, -49.374331387606546, 69.1869693618585),
              str, 20, 0.5, years[b], "Jakobshavn", plot = True)
        print(years[b] + " done")
    if b == 10:
        print("finished!")
        break














