import os
import functions
import matplotlib.pyplot as plt
import numpy as np
from patchify import patchify
from PIL import Image
## get data
# load data
#times = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
times = ["2013"]
# change path
os.chdir("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/datasets/jakobshavn")
path = os.getcwd()
os.chdir(path)

print("Begin loading data")
# read
fullData = []
for i in range(len(times)):
    helper = functions.openData(times[i])
    fullData.append(helper)

d = [item for sublist in fullData for item in sublist]
print(len(d))

## add ndsi masks

d = functions.NDSI(d, threshold = 0.3)


for i in range(len(d)):# rgb
    #img = functions.createImage(d[i][1][1:4,:,:], 0.4)
    img = d[i][1][9,:,:]
    plt.imshow(img)
    plt.show()
    

"""
testImg = d[2][1][:, :, :]


patch = functions.createPatches(testImg, (400, 400), 3, [0,500, 0,500])

testImg1 = d[3][1][:, :, :]

patch1 = functions.createPatches(testImg1, (400,400), 3, [0,500, 0,500])
print(patch1.shape)
plt.imshow(patch[2,2,:,:])
plt.show()

plt.imshow(patch1[2,2,:,:])
plt.show()

"""



















"""
import imageio
imageio.imwrite('filename.jpg', rgb)

# print as pathces
image = Image.open("filename.jpg")  # for example (3456, 5184, 3)
image = np.asarray(image)
patches = patchify(image, (200, 200, 3), step=200)
print(patches.shape)  # (6, 10, 1, 512, 512, 3)

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0]
        patch = Image.fromarray(patch)
        num = i * patches.shape[1] + j
        patch.save(f"patch_{num}.jpg")
"""