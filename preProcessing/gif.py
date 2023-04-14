import numpy as np
import imageio
import functions
import os


# aligned gif
# Create a list of numpy arrays
arr_list = []
for i in range(34):
    img = functions.openData(
        "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/" + str(i))
    img = np.nan_to_num(img)
    arr_list.append(img)



# Create a GIF from the numpy arrays
with imageio.get_writer("animation.gif", mode="I") as writer:
    for arr in arr_list:
        writer.append_data(arr)
"""
# unaligned gif
ROI = [0,800,0,800]
years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati"
os.chdir(path)
d = functions.loadData(path, years)

# check for the correct months to make data stationary -> summer data
l = []
counterImg = 0
usedMonths = []
months = [[i, i + 1, i + 2] for i in range(1, 12, 3) if i + 2 <= 12]

for y in np.arange(2013, 2022, 1):  # year
    # for m in np.arange(1,13,1): # month
    for [m, t, h] in months:
        imgAcc = np.zeros((2, ROI[1] - ROI[0], ROI[3] - ROI[2]))
        month = 0
        for i in range(len(d)):
            if (((functions.convertDatetoVector(d[i][0])[1].item() == m) or (functions.convertDatetoVector(d[i][0])[1].item() == t) or (
                    functions.convertDatetoVector(d[i][0])[1].item() == h)) and (functions.convertDatetoVector(d[i][0])[2].item() == y)):
                # count months
                month += 1

                ## get roi and apply kernel
                img = d[i][1][[2,5], ROI[0] : ROI[1] , ROI[2] : ROI[3]]


                # average
                if month == 1:
                    imgAcc += img
                if month > 1:
                    imgAcc = (imgAcc + img) / 2  # average

        # apply NDSI here
        # add snow mask to average image
        threshold = 0.3
        NDSI = np.divide(np.subtract(imgAcc[0, :, :], imgAcc[1, :, :]),
                         np.add(imgAcc[0, :, :], imgAcc[1, :, :]))
        # nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0) ## leave in case necessary to use in future
        snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
        l.append(snow)
        usedMonths.append(np.array([[m, y]]))

    print("averaging of year: ", y, " done")


l = [np.nan_to_num(l[i]) for i in range(len(l))]
print(len(l))
# Create a GIF from the numpy arrays
with imageio.get_writer("animation.gif", mode="I") as writer:
    for arr in l:
        writer.append_data(arr)
        
        
"""
