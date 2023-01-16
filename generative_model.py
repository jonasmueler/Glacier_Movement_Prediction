import functions 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import datetime
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor



def sharpening(img, alpha):
    """
    
    img: 2d np.array
        input image 
    alpha: int 
        alpha value used for sharpening

    """
    
    
    blurred_f = ndimage.gaussian_filter(img, 3)

    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

    alpha = 25
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    
    return sharpened

def getSnowPixel(imgList):
    """
    counts number of snow pixel for snow mask

    Parameters
    ----------
    imgList : list of 2d np.array
        images over time.

    Returns
    -------
    snowPix : list of int
        non zero pixel.

    """
    snowPix = list(map(lambda x: np.sum(np.ndarray.flatten(x) > 0), imgList))
    return snowPix

def moving_avg(x, n):
    """
    
    calculate moving average
    
    x : array
    n : int
        elements in average.

    Returns
    -------
    array
        smoothed values.

    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)


## get data
# load data
times = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]

# change path  
os.chdir("D:/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo")
path = os.getcwd() + "/images"
os.chdir(path)
    
print("Begin loading data")
#read 
fullData = []
for i in range(len(times)):
    helper = functions.openData(times[i])
    fullData.append(helper)
    
d = [item for sublist in fullData for item in sublist]
    
print("loading data finished")

def imageGeneration(path, gamma, band, beginning, end, deltaDays, alpha, data):
    """
    
    interpolates dates between images with synthetic images created from a gaussian 
    process prior 
    
    path : str
        path to images.
    gamma : float
        controls amount of last image used for current image.
    band : int
        which band is used.
    beginning : int
        start year.
    end : int
        end year.
    deltaDays : int
        delay between interpolations in days.
    alpha : int
        alpha value used for image sharpening.

    Returns
    -------
    d : list of date np.array and image np.array
        interpolated image series.

    """

    
    # train gaussian process environment prior for image generation
    ## train data 
    times = []
    for i in range(len(data)):
        helper = str(data[i][0])
        year = int(helper[0:4])
        month = int(helper[5:7])
        day = int(helper[8:10])
        res = np.array([year, month, day])
        times.append(res)
        
    dates = np.vstack(times)
    
    targets = np.vstack([np.ndarray.flatten(data[x][1][0:888,0:953,band]) for x in range(len(data))])

    #train 
    print("start model training")
    gpr = GaussianProcessRegressor()
    #gpr =  MLPRegressor(hidden_layer_sizes = (10,20,40,50,50,), activation = "relu", verbose = True,
                       #max_iter = 10, 
                      #batch_size= 5)
    gpr.fit(dates, targets)
    print("training finished")
    
    # image interpolation
    # create datetime vectors 

    dt = datetime.datetime(beginning, 1, 1)
    end = datetime.datetime(end, 1, 1)
    step = datetime.timedelta(days=deltaDays)

    result = []
    while dt < end:
        result.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
        dt += step
    
    # convert to vector 
    timesInt = []
    years = []
    for i in range(len(result)):
        helper = str(result[i])
        year = int(helper[0:4])
        month = int(helper[5:7])
        day = int(helper[8:10])
        res = np.array([year, month, day])
        timesInt.append(res)
        years.append(year)
        
    datesInt = np.vstack(timesInt)
    
    print("start creation of synthetic data")
    #feed into gaussian process, create synthetic data
    d = []
    codesOrg = np.apply_along_axis(lambda x: str(x[0]) + str(x[1]) + str(x[2]),1, dates).tolist()
    codesInt = np.apply_along_axis(lambda x: str(x[0]) + str(x[1]) + str(x[2]),1, datesInt).tolist()
    for i in range(len(datesInt)):
        if (codesInt[i] in codesOrg) == False: 
            res = gpr.predict(np.array(datesInt[i]).reshape(1, -1)).reshape(888, 953)
            d.append(res)
        elif (codesInt[i] in codesOrg) == True:
            ind = codesOrg.index(codesInt[i])
            res = data[ind][1][:,:,band]
            d.append(res)
        if i%200 == 0:
            print("Image ", i,"of ", len(datesInt), " created")
    
    # apply moving average over data with interpolated synthetic images again
    #img = d[0]
    #for i in range(len(data)-1):
    #    curImage = d[i+1]
    #    img = gamma*img + (1-gamma)*curImage
    #    d[i+1] = sharpening(img, alpha=alpha) 
    #print("process finished")
    
    return [datesInt, d]




p = "D:/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/images"
res = imageGeneration(p, 0.1, 1, 2012, 2023, 1, 40, d)  

# plot iimages
plt.figure(figsize=(80, 80)) 
#plt.imshow(res[90]) 
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(res[i]) 
    
# plot yearly/monthly snow pixel amount 
times = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
times = list(map(lambda x: int(x), times))

means = []
for i in range(len(times)):
    #helper = [res[1][x] for x in range(len(res)) if res[0][x,0] == times[i]]
    helper = []
    
    for b in range(len(res[1])):
        if res[0][b,1] == times[i]:
            r = np.sum(np.ndarray.flatten(res[1][b]))
            helper.append(r)
    means.append(np.mean(helper))
    
plt.plot(means)
   

    
# plot images in summer

times = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
times = list(map(lambda x: int(x), times))
months = [6]
summer = []
for i in range(len(months)):   
    for b in range(len(res[1])):
        if res[0][b,1] == months[i]:
            r = res[1][b]
            plt.imshow(r)
            plt.show()
            
# plot pixel in summer over time 
#months = [5,6,7,8,9]
months = [7,8,9]
summer = []
for i in range(len(months)):   
    for b in range(len(res[1])):
        if res[0][b,1] == months[i]:
            r = np.sum(np.ndarray.flatten(res[1][b]) > 0)
            
            summer.append(r)

plt.plot(moving_avg(summer, 100))
plt.plot(moving_avg(summer, 200))
plt.plot(moving_avg(summer, 300))
plt.plot(moving_avg(summer, 40))
plt.plot(moving_avg(summer, 60))
plt.plot(moving_avg(summer, 80))      
            
    
## byrd glacier
p = "D:/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/images"
r = imageGeneration(p, 0.1, 0, 2012, 2023, 1, 40, d) 
g = imageGeneration(p, 0.1, 1, 2012, 2023, 1, 40, d) 
b = imageGeneration(p, 0.1, 2, 2012, 2023, 1, 40, d) 



    
for i in range(len(d)):
    #print(d[i][1][0:888,0:953,0].shape)
    img = np.

import gc

gc.collect()



