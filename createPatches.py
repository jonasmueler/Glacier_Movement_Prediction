import os
import functions
import matplotlib.pyplot as plt
import pickle
from collections import Counter

########### first check images with plot, then save patched pickle files #######################
glacier = "parbati"

if glacier == "aletsch":
    years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    goodImgs = [[0,2,5,7,8], [1,2,3,4,7,8], [1,2,3,4,5,8,10,12,14],
                [0,2,7,9,11,12], [2,3,4,5,6,7,8], [2,5,10,11,12], [2,5,6,7,9,11], [1,2,3,4,5,7,10,12], [1,3,4,5,6,7,8,9,10,12,15]]
    for year in enumerate(years):

        path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn"
        os.chdir(path)
        d = functions.loadData(path, [year[1]])

        ##################### plotting ########################
        """
        # find roi and good images
        for i in range(len(d)):# rgb
            img = functions.createImage(d[i][1][1:4,:,:], 0.4)
            #img = d[i][1][:,:, 9]
            plt.imshow(img)
            plt.show()
        """
        #######################################################
        # get good images
        d = [d[x] for x in goodImgs[year[0]]]
        # check for the correct months to make data stationary -> summer data
        l = []
        months = []
        for i in range(len(d)):
            months.append(functions.convertDatetoVector(d[i][0])[1].item())
            if (functions.convertDatetoVector(d[i][0])[1] == 4) or (functions.convertDatetoVector(d[i][0])[1] == 5) or (functions.convertDatetoVector(d[i][0])[1] == 6) or (functions.convertDatetoVector(d[i][0])[1] == 7):
                l.append(d[i])

        months = dict(Counter(months))
        print(months)

        d = functions.automatePatching(l, 50, 40, [50, 650, 100, 700] , applyKernel=True)

        os.makedirs(path + "/patched", exist_ok = True)
        os.chdir(path + "/patched")

        with open(year[1], "wb") as fp:  # Pickling
            pickle.dump(d, fp)
        print("data saved!")
        print("year: ", year, " done" )





if glacier == "helheim":
    years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    goodImgs = [[2, 3, 4, 5, 6, 7, 8, 10],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24], [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [0, 2, 3, 4, 7, 8, 9, 10, 11, 12],
        [0, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21],
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
    for year in enumerate(years):

        path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim"
        os.chdir(path)
        d = functions.loadData(path, [year[1]])

        ##################### plotting ########################
        """
        # find roi and good images
        for i in range(len(d)):# rgb
            img = functions.createImage(d[i][1][1:4,:,:], 0.4)
            #img = d[i][1][:,:, 9]
            plt.imshow(img)
            plt.show()
        """
        #######################################################
        # check for the correct months to make data stationary -> summer data
        l = []
        months = []
        for i in range(len(d)):
            months.append(functions.convertDatetoVector(d[i][0])[1].item())
            if (functions.convertDatetoVector(d[i][0])[1] == 3) or (functions.convertDatetoVector(d[i][0])[1] == 4):
                l.append(d[i])

        months = dict(Counter(months))
        print(months)


        d = functions.automatePatching(l, 50, 40, [100, 400, 200, 500] , applyKernel=True)

        os.makedirs(path + "/patched", exist_ok = True)
        os.chdir(path + "/patched")

        with open(year[1], "wb") as fp:  # Pickling
            pickle.dump(d, fp)
        print("data saved!")
        print("year: ", year, " done" )


if glacier == "Jakobshavn":
    years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    goodImgs = [[0, 1, 2,3,4, 6, 8], [1, 3, 4,  7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29]
, [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13,14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
, [1, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
, [1, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28]
, [0, 1, 2, 3, 4, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21]
, [1, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]#31]
, [0,1,2,3,5,6,8,9,10,11,12,13,14,15,16, 17, 18]
, [0,1,2,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19]]

    for year in enumerate(years):

        path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jakobshavn"
        os.chdir(path)
        d = functions.loadData(path, [year[1]])

        ##################### plotting ########################
        """
        # find roi and good images
        for i in range(len(d)):# rgb
            img = functions.createImage(d[i][1][1:4,:,:], 0.4)
            #img = d[i][1][:,:, 9]
            plt.imshow(img)
            plt.show()
        """
        #######################################################
        # get good images
        d = [d[x] for x in goodImgs[year[0]]]

        # check for the correct months to make data stationary -> summer data
        l = []
        months = []
        for i in range(len(d)):
            months.append(functions.convertDatetoVector(d[i][0])[1].item())
            if (functions.convertDatetoVector(d[i][0])[1] == 3) or (functions.convertDatetoVector(d[i][0])[1] == 4) or (functions.convertDatetoVector(d[i][0])[1] == 5):
                l.append(d[i])

        months = dict(Counter(months))
        print(months)


        d = functions.automatePatching(l, 50, 40, [0, 300, 0, 300] , applyKernel=True)

        os.makedirs(path + "/patched", exist_ok = True)
        os.chdir(path + "/patched")

        with open(year[1], "wb") as fp:  # Pickling
            pickle.dump(d, fp)
        print("data saved!")
        print("year: ", year, " done" )#



if glacier == "parbati":
    years = ["2013", "2014" , "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati"
    os.chdir(path)
    d = functions.loadData(path, years)

    # check for the correct months to make data stationary -> summer data
    d = functions.monthlyAverageScenes(d, [0,800,0,800], True)

    ##################### plotting ########################
    """
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








