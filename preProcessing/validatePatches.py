import os

import functions
import matplotlib.pyplot as plt
import cv2
import numpy as np

# data
#img = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/0")
#img1 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/1")
#img2 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/3")
#img3 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/4")

# predictions
threshold = 0.3
# LSTMattentionsmall
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/results/LSTMAttentionSmall/modelPredictions/parvati/1/predictions"
# Load the two PDF files as images
images1 = functions.openData(path + "/0")[0,:,:]
images2 = functions.openData(path + "/1")[0,:,:]
images3 = functions.openData(path + "/2")[0,:,:]
images4 = functions.openData(path + "/3")[0,:,:]

# threshold again
images1 = np.ma.masked_where(images1 < threshold, images1).filled(0)
images2 = np.ma.masked_where(images2 < threshold, images2).filled(0)
images3 = np.ma.masked_where(images3 < threshold, images3).filled(0)
images4 = np.ma.masked_where(images4 < threshold, images4).filled(0)


# global
gridsize = 10
params = {"scale": 0.5, "levels": 6, "window": 40, "iterations": 10, "poly": 5, "sigma": 1.1}
color = "olive"
# Create a figure with a single row and five columns
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(26, 18))
# Turn off axis for all subplots
for ax in axs.flat:
    ax.axis('off')

# Plot the first image on the first column
axs[0].imshow(images1, cmap='gray')
#axs[0].set_title('Image 1')

# Plot the second image on the second column
frame1 = images1.astype(np.float32)
frame2 = images2.astype(np.float32)
frame1= cv2.merge([frame1,frame1, frame1])
frame2 = cv2.merge([frame2, frame2, frame2])


# Convert frames to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Compute optical flow using Lucas-Kanade method
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, params["scale"], params["levels"], params["window"], params["iterations"], params["poly"], params["sigma"], 0)
h, w = prev_gray.shape
x, y = np.meshgrid(np.arange(0, w, gridsize), np.arange(0, h, gridsize))
x_flow = flow[..., 0][::gridsize, ::gridsize]
y_flow = flow[..., 1][::gridsize, ::gridsize]
axs[1].quiver(x, y, x_flow, y_flow, color = color, width = 0.0016)
axs[1].imshow(images2, cmap='gray')
#axs[1].set_title('Image 2')

plt.tight_layout()
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots"
os.chdir(path)
plt.savefig("parbatiFlowPredictions.pdf", dpi = 1000)
plt.show()

"""
# Plot the third image on the third column
frame1 = img1.astype(np.float32)
frame2 = img2.astype(np.float32)
frame1= cv2.merge([frame1,frame1, frame1])
frame2 = cv2.merge([frame2, frame2, frame2])


# Convert frames to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Compute optical flow using Lucas-Kanade method
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, params["scale"], params["levels"], params["window"], params["iterations"], params["poly"], params["sigma"], 0)
h, w = prev_gray.shape
x, y = np.meshgrid(np.arange(0, w, gridsize), np.arange(0, h, gridsize))
x_flow = flow[..., 0][::gridsize, ::gridsize]
y_flow = flow[..., 1][::gridsize, ::gridsize]
axs[1, 0].quiver(x, y, x_flow, y_flow, color = color, width = 0.0014)
axs[1, 0].imshow(img2, cmap='gray')
#axs[2].set_title('Image 3')


# Plot the fourth image on the fourth column
frame1 = img2.astype(np.float32)
frame2 = img3.astype(np.float32)
frame1= cv2.merge([frame1,frame1, frame1])
frame2 = cv2.merge([frame2, frame2, frame2])


# Convert frames to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Compute optical flow using Lucas-Kanade method
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, params["scale"], params["levels"], params["window"], params["iterations"], params["poly"], params["sigma"], 0)
h, w = prev_gray.shape
x, y = np.meshgrid(np.arange(0, w, gridsize), np.arange(0, h, gridsize))
x_flow = flow[..., 0][::gridsize, ::gridsize]
y_flow = flow[..., 1][::gridsize, ::gridsize]
axs[1, 1].quiver(x, y, x_flow, y_flow, color = color, width = 0.0014)
axs[1, 1].imshow(img3, cmap='gray')
#axs[3].set_title('Image 4')


# Show the plot
plt.tight_layout()
plt.show()
"""
"""
################################# plot scenes ###################################
image = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/0")
image1 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/1")
image2 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/2")
image3 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/3")

image4= functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/4")
image5 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/5")
image6 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/6")
image7 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/7")

image8= functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/8")
image9 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/9")
image10 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/10")
image11 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/11")




# create a figure object with 3 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 20))

# remove the axes and ticks from all subplots
for ax in axs.flat:
    ax.axis('off')

# plot the images in the subplots
axs[0, 0].imshow(image, cmap= "gray")
axs[0, 1].imshow(image1, cmap="gray")
axs[0, 2].imshow(image2, cmap="gray")
axs[0, 3].imshow(image3, cmap="gray")
axs[1, 0].imshow(image4, cmap="gray")
axs[1, 1].imshow(image5, cmap="gray")
axs[1, 2].imshow(image6, cmap="gray")
axs[1, 3].imshow(image7, cmap="gray")
axs[2, 0].imshow(image8, cmap="gray")
axs[2, 1].imshow(image9, cmap="gray")
axs[2, 2].imshow(image10, cmap="gray")
axs[2, 3].imshow(image11, cmap="gray")

# display the figure
plt.savefig("scenesYear.pdf", dpi = 1000)
plt.tight_layout()
plt.show()
"""