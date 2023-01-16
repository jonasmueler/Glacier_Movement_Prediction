import functions
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np
year = "2015"


d = functions.loadData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/jungfrau_aletsch_beitschhorn", [year])

## add ndsi masks
d = functions.NDSI(d, threshold = 0.3)
print(d[1][0])
# find roi and good images
#d[0][1][1,:,:] = functions.applyToImage(d[22][1][1,:,:])
#d[0][1][2,:,:] = functions.applyToImage(d[22][1][2,:,:])
#d[0][1][3,:,:] = functions.applyToImage(d[22][1][3,:,:])
#img = functions.createImage(d[1][1][1:4,:,:], 0.4)
img = (d[1][1][9,:,:] * 255.99).astype("uint8")
#img = np.transpose(img, (2, 0,1))
print(img.shape)
#img = d[i][1][:,:, 9]
plt.imshow(img)
#import matplotlib.image
#matplotlib.image.imsave('name.png', img, cmap = "binary")
import imageio
imageio.imwrite('aletschSnow.jpg', img)

plt.show()
