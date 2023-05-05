import functions
import matplotlib.pyplot as plt
import cv2
import numpy as np

# load data
data = []
for i in range(104):
    img1 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/parbatiScenes/"
                          + str(i))
    data.append(img1)


data = functions.aligneOverTime(data)

for i in range(len(data)):
    plt.imshow(data[i])
    plt.show()

