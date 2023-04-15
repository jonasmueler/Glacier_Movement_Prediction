import os

import functions
import matplotlib.pyplot as plt
import cv2
import numpy as np

img = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/0")
img1 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/1")
img2 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/3")
img3 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenes/images/4")


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
axs[0].imshow(img, cmap='gray')
#axs[0].set_title('Image 1')

# Plot the second image on the second column
frame1 = img.astype(np.float32)
frame2 = img1.astype(np.float32)
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
axs[1].imshow(img1, cmap='gray')
#axs[1].set_title('Image 2')

plt.tight_layout()
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots"
os.chdir(path)
#plt.savefig("parbatiFlow.pdf", dpi = 1000)
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
