import os
import pickle
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import functions


# aletsch
# targets
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/experiment3/results/LSTMAttentionSmallMixedData/modelPredictions/aletsch/0/targets"
threshold = 0.3
# Load the two PDF files as images
images1 = functions.openData(path + "/0")[0,:,:]
images2 = functions.openData(path + "/1")[0,:,:]
images3 = functions.openData(path + "/2")[0,:,:]
images4 = functions.openData(path + "/3")[0,:,:]

# LSTMattentionsmall
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/experiment3/results/LSTMAttentionSmallMixedData/modelPredictions/aletsch/0/predictions"
# Load the two PDF files as images
images5 = functions.openData(path + "/0")[0,:,:]
images6 = functions.openData(path + "/1")[0,:,:]
images7 = functions.openData(path + "/2")[0,:,:]
images8 = functions.openData(path + "/3")[0,:,:]

# threshold again
images5 = np.ma.masked_where(images5 < threshold, images5).filled(0)
images6 = np.ma.masked_where(images6 < threshold, images6).filled(0)
images7 = np.ma.masked_where(images7 < threshold, images7).filled(0)
images8 = np.ma.masked_where(images8 < threshold, images8).filled(0)


# parvati
# targets
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/experiment3/results/LSTMAttentionSmallMixedData/modelPredictions/parvati/1/targets"

images9 = functions.openData(path + "/0")[0,:,:]
images10 = functions.openData(path + "/1")[0,:,:]
images11 = functions.openData(path + "/2")[0,:,:]
images12 = functions.openData(path + "/3")[0,:,:]


# model preds
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/experiment3/results/LSTMAttentionSmallMixedData/modelPredictions/parvati/1/predictions"

images13 = functions.openData(path + "/0")[0,:,:]
images14 = functions.openData(path + "/1")[0,:,:]
images15 = functions.openData(path + "/2")[0,:,:]
images16 = functions.openData(path + "/3")[0,:,:]

# threshold again
images13 = np.ma.masked_where(images13 < threshold, images13).filled(0)
images14 = np.ma.masked_where(images14 < threshold, images14).filled(0)
images15 = np.ma.masked_where(images15 < threshold, images15).filled(0)
images16 = np.ma.masked_where(images16 < threshold, images16).filled(0)


# Create a figure with two subplots
fig, axs = plt.subplots(4, 4, figsize=(60, 70))
# Adjust the space between subplots
fig.subplots_adjust(hspace=0.01, wspace = 0.01)

fontsize = 120

axs[0,0].imshow(images1, cmap = "gray")
axs[0,0].set_title('a)', fontsize = fontsize)
axs[0,0].set_axis_off()


axs[0,1].imshow(images2, cmap = "gray")

axs[0,1].set_axis_off()

axs[0,2].imshow(images3, cmap = "gray")

axs[0,2].set_axis_off()

axs[0,3].imshow(images4, cmap = "gray")

axs[0,3].set_axis_off()

axs[1,0].imshow(images5, cmap = "gray")
axs[1,0].set_title('b)', fontsize = fontsize)
axs[1,0].set_axis_off()

axs[1,1].imshow(images6, cmap = "gray")
#axs[0].set_title('First PDF')
axs[1,1].set_axis_off()

axs[1,2].imshow(images7, cmap = "gray")
#axs[0].set_title('First PDF')
axs[1,2].set_axis_off()

axs[1,3].imshow(images8, cmap = "gray")
#axs[0].set_title('First PDF')
axs[1,3].set_axis_off()

axs[2,0].imshow(images9, cmap = "gray")
axs[2,0].set_title('c)', fontsize = fontsize)
axs[2,0].set_axis_off()

axs[2,1].imshow(images10, cmap = "gray")

axs[2,1].set_axis_off()

axs[2,2].imshow(images11, cmap = "gray")

axs[2,2].set_axis_off()

axs[2,3].imshow(images12, cmap = "gray")

axs[2,3].set_axis_off()

axs[3,0].imshow(images13, cmap = "gray")
axs[3,0].set_title('d)', fontsize = fontsize)
axs[3,0].set_axis_off()

axs[3,1].imshow(images14, cmap = "gray")
#axs[0].set_title('First PDF')
axs[3,1].set_axis_off()

axs[3,2].imshow(images15, cmap = "gray")
#axs[0].set_title('First PDF')
axs[3,2].set_axis_off()

axs[3,3].imshow(images16, cmap = "gray")
#axs[0].set_title('First PDF')
axs[3,3].set_axis_off()


"""
# Loop over the rows and columns of the axes
for row in range(0, 5):
    for col in range(4):
        if col == 3:
            # Add a colorbar to the fourth plot in each row
            im = axs[row, col].imshow(eval(f"images{4 * row + col + 1}"), cmap = "gray")
            axs[row, col].set_axis_off()
            cax = fig.add_axes([0.9, axs[row, col].get_position().y0, 0.01, axs[row, col].get_position().height])
            fig.colorbar(im, cax=cax)


        else:
            axs[row, col].imshow(eval(f"images{4 * row + col + 1}"), cmap = "gray")
            axs[row, col].set_axis_off()
"""
for row in range(0, 4):
    for col in range(4):
        if col == 3:
            # Add a colorbar to the fourth plot in each row
            im = axs[row, col].imshow(eval(f"images{4 * row + col + 1}"), cmap = "gray")
            axs[row, col].set_axis_off()
            cax = fig.add_axes([0.9, axs[row, col].get_position().y0, 0.01, axs[row, col].get_position().height])
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=60) # adjust the value of labelsize as per your requirement
        else:
            axs[row, col].imshow(eval(f"images{4 * row + col + 1}"), cmap = "gray")
            axs[row, col].set_axis_off()

# Show the plot

#os.chdir("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots")


#plt.tight_layout()
plt.savefig("ScenePredictionsMixedTrain.pdf", dpi = 200)
plt.show()