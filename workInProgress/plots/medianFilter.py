import functions
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment2/results/LSTMAttentionSmallAletsch/modelPredictions/aletsch/0/predictions/0"

img = functions.openData(path)[0,:,:]


# apply a median filter of size 3x3
filtered_image = ndimage.median_filter(img, size=3)

threshold = 0.3
filtered_image = np.ma.masked_where(filtered_image < threshold, filtered_image).filled(0)

plt.imshow(filtered_image, cmap = "gray")
plt.show()

