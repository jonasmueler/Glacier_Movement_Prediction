import os
import functions
import matplotlib.pyplot as plt
import pickle

########### first check images with plot, then save patched pickle files #######################
year = "2013"

path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn"
os.chdir(path)
d = functions.loadData(path, [year])

## add ndsi masks
d = functions.NDSI(d, threshold = 0.3)


# find roi and good images
for i in range(len(d)):# rgb
    img = functions.createImage(d[i][1][1:4,:,:], 0.4)
    #img = d[i][1][:,:, 9]
    plt.imshow(img)
    plt.show()

"""
# 2013
#goodImg = [2, 3, 4, 5, 6, 7, 8, 10]
# 2014:
#goodImg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# 2015:
#goodImg = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
# 2016
#goodImg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]
# 2017
#goodImg = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# 2018
#goodImg = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12]
# 2019
#goodImg = [0, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]
# 2020
#goodImg = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 2021
#goodImg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
d = [d[i] for i in goodImg]

d = functions.automatePatching(d, 50, 20, [100, 400, 200, 500], applyKernel=True)

os.makedirs(path + "/patched", exist_ok = True)
os.chdir(path + "/patched")


with open(year, "wb") as fp:  # Pickling
    pickle.dump(d, fp)
print("data saved!")

"""




