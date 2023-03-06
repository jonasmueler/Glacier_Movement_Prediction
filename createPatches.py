import os
import functions
import matplotlib.pyplot as plt
import pickle

########### first check images with plot, then save patched pickle files #######################
# start from 2017 again
year = "2021"

path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn"
os.chdir(path)
d = functions.loadData(path, [year])



## add ndsi masks
#d = functions.NDSI(d, threshold = 0.3)

"""
# find roi and good images
for i in range(len(d)):# rgb
    img = functions.createImage(d[i][1][1:4,:,:], 0.4)
    #img = d[i][1][:,:, 9]
    plt.imshow(img)
    plt.show()
"""

# 2013
#goodImg = [0,2,5,7,8]
# 2014:
#goodImg = [1,2,3,4,7,8]
# 2015:
#goodImg = [1,2,3,4,5,8,10,12,14]
# 2016
#goodImg = [0,2,7,9,11,12]
# 2017
#goodImg = [2,3,4,5,6,7,8]
# 2018
#goodImg = [2,5,10,11,12]
# 2019
#goodImg = [2,5,6,7,9,11]
# 2020
#goodImg = [1,2,3,4,5,7,10,12]
# 2021
#goodImg = [1,3,4,5,6,7,8,9,10,12,15]




d = [d[i] for i in goodImg]

d = functions.automatePatching(d, 50, 40, [50, 650, 100, 700] , applyKernel=True)

os.makedirs(path + "/patched", exist_ok = True)
os.chdir(path + "/patched")


with open(year, "wb") as fp:  # Pickling
    pickle.dump(d, fp)
print("data saved!")






