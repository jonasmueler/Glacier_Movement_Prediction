import os
import functions
import matplotlib.pyplot as plt
import pickle
from collections import Counter

path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati" # change here

########### first check images with plot, then save patched pickle files #######################
glacier = "parvati"

if glacier == "aletsch":
    years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    goodImgs = [[0,2,5,7,8], [1,2,3,4,7,8], [1,2,3,4,5,8,10,12,14],
                [0,2,7,9,11,12], [2,3,4,5,6,7,8], [2,5,10,11,12], [2,5,6,7,9,11], [1,2,3,4,5,7,10,12], [1,3,4,5,6,7,8,9,10,12,15]]

    os.chdir(path)
    d = []
    for i in range(len(years)):
        helper = functions.loadData(path, [years[i]])
        # get good images
        helper = [helper[x] for x in goodImgs[i]]
        d += helper

    d = functions.monthlyAverageScenesEnCC(d, [50, 650, 100, 700], True)
    """
    ##################### plotting ########################
    print(len(d))
    # find roi and good images
    for i in range(len(d)):  # rgb
        #img = functions.createImage(d[i][1][1:4, :, :], 0.4)
        img = d[i]
        plt.imshow(img)
        plt.show()
    """
    d = d[0:19]
    d = functions.automatePatching(d, 50, 10)

    os.makedirs(path + "/patched", exist_ok=True)
    os.chdir(path + "/patched")

    with open("AletschPatched", "wb") as fp:  # Pickling
        pickle.dump(d, fp)
    print("data saved!")



if glacier == "parvati":
    years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    os.chdir(path)
    d = functions.loadData(path, years)

    # check for the correct months to make data stationary -> summer data
    d = functions.monthlyAverageScenesEnCC(d, [0,800,0,800], True)
    """
    ##################### plotting ########################
    print(len(d))
    # find roi and good images
    for i in range(len(d)):  # rgb
        #img = functions.createImage(d[i][1][1:4, :, :], 0.4)
        img = d[i]
        plt.imshow(img)
        plt.show()
    """
    d = functions.automatePatching(d, 50, 10)

    os.makedirs(path + "/patched", exist_ok = True)
    os.chdir(path + "/patched")

    with open("parbatiPatched", "wb") as fp:  # Pickling
        pickle.dump(d, fp)
    print("data saved!")










