import os
import functions
import matplotlib.pyplot as plt
import cv2
import numpy as np

# data
images1 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/results/LSTMAttentionSmall/modelPredictions/parvati/1/targets/0")[0,:,:]
images2 = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/results/LSTMAttentionSmall/modelPredictions/parvati/1/targets/1")[0,:,:]

# predictions
threshold = 0.3

# LSTMattentionsmall
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/results/LSTMAttentionSmall/modelPredictions/parvati/1/predictions"
# Load data
images1Prediction = functions.openData(path + "/0")[0,:,:]
images2Prediction = functions.openData(path + "/1")[0,:,:]


# threshold again
images1Prediction = np.ma.masked_where(images1Prediction < threshold, images1Prediction).filled(0)
images2Prediction= np.ma.masked_where(images2Prediction< threshold, images2Prediction).filled(0)

# global
gridsize = 10
params = {"scale": 0.5, "levels": 6, "window": 60, "iterations": 10, "poly": 5, "sigma": 1.1}
color = "olive"
# Create a figure with a single row and 2 columns
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))

# Turn off axis for all subplots
for ax in axs.flat:
    ax.axis('off')

# Adjust the space between subplots
fig.subplots_adjust(hspace=0.01, wspace = 0.01)


# Plot the second image on the second column
frame1 = images1.astype(np.float32)
frame2 = images2.astype(np.float32)
frame1= cv2.merge([frame1,frame1, frame1])
frame2 = cv2.merge([frame2, frame2, frame2])


# Convert frames to grayscale
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Compute optical flow
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, params["scale"], params["levels"], params["window"], params["iterations"], params["poly"], params["sigma"], 0)
h, w = prev_gray.shape
x, y = np.meshgrid(np.arange(0, w, gridsize), np.arange(0, h, gridsize))
x_flow = flow[..., 0][::gridsize, ::gridsize]
y_flow = flow[..., 1][::gridsize, ::gridsize]

axs[0].quiver(x, y, x_flow, y_flow, color = color, width = 0.002) # 0.0016
axs[0].imshow(images2, cmap='gray')

# Plot the third image on the third column
frame1 = images1Prediction.astype(np.float32)
frame2 = images2Prediction.astype(np.float32)
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
axs[1].quiver(x, y, x_flow, y_flow, color = color, width = 0.002)
axs[1].imshow(images2Prediction, cmap='gray')


#plt.tight_layout()
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots"
os.chdir(path)

for col in range(2):
    if col == 1:
        # Add a colorbar to the fourth plot in each row
        im = axs[col].imshow(images2Prediction, cmap="gray")
        axs[col].set_axis_off()
        cax = fig.add_axes([0.9, axs[col].get_position().y0, 0.01, axs[col].get_position().height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=30)  # adjust the value of labelsize as per your requirement
plt.savefig("parbatiFlowPredictions.pdf", dpi = 300)
plt.show()


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
