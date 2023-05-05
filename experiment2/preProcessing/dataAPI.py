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

# global; change path here
pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets"

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

    # save on hard drive with pickling
    # create folder

    pathOrigin = pathOrigin + "/" + glacierName
    os.makedirs(pathOrigin, exist_ok = True)
    os.chdir(pathOrigin)

    """
    # change folder
    os.makedirs(pathOrigin + "/" + "examples", exist_ok = True)
    os.chdir(pathOrigin + "/" + "examples")
    
    #plot as check in RGB
    if plot: # sanity check
        for i in range(4):
            rgb = functions.createImage(d[i][1][1:4, 0:800, 0:800], 0.40)
            plt.imshow(rgb)
            plt.axis("off")

            # clear date string
            name = str(d[i][0])
            name = name.replace(":", "-")
            name = name.replace(".", "-")

            # save
            plt.savefig(name + ".pdf", dpi = 1000)


            plt.show()
        for i in range(len(d)): #rgb
            rgb = functions.createImage(d[i][1][1:4,:,:], 0.40)
            plt.subplot(round(len(d)/2)+1, 2, i + 1)
            plt.imshow(rgb)
        plt.show()
    """
    # change back to origin folder
    os.chdir(pathOrigin)

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
        string = years[b] + "-01-01/" + years[b+1] + "-01-01"
        API((7.889556884765626, 46.382464893261165, 8.22257995605469, 46.57160668424229), # add coordinates here
              string, 20, 0.5, years[b], "Jungfrau-Aletsch-Bietschhorn", plot = True)
        print(years[b] + " done")
    if b == 10:
        print("finished!")
        break














