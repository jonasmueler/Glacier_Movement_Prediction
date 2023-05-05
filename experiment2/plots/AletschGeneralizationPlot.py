import os
import pickle
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import functions
from scipy import ndimage


# targets
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment2/results/LSTMAttentionSmall/modelPredictions/aletsch/0/targets"

threshold = 0.3
# Load the two PDF files as images
images1 = functions.openData(path + "/0")[0,:,:]
images2 = functions.openData(path + "/1")[0,:,:]
images3 = functions.openData(path + "/2")[0,:,:]
images4 = functions.openData(path + "/3")[0,:,:]

# LSTMattentionsmall
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment2/results/LSTMAttentionSmall/modelPredictions/aletsch/0/predictions"
# Load the two PDF files as images
images5 = functions.openData(path + "/0")[0,:,:]
images6 = functions.openData(path + "/1")[0,:,:]
images7 = functions.openData(path + "/2")[0,:,:]
images8 = functions.openData(path + "/3")[0,:,:]

# threshold again
images5 = ndimage.median_filter(np.ma.masked_where(images5 < threshold, images5).filled(0), size= 3)
images6 = ndimage.median_filter(np.ma.masked_where(images6 < threshold, images6).filled(0), size= 3)
images7 = ndimage.median_filter(np.ma.masked_where(images7 < threshold, images7).filled(0), size= 3)
images8 = ndimage.median_filter(np.ma.masked_where(images8 < threshold, images8).filled(0), size= 3)

# LSTMattentionsmall Aletsch trained
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment2/results/LSTMAttentionSmallAletsch/modelPredictions/aletsch/0/predictions"
# Load the two PDF files as images
images9 = functions.openData(path + "/0")[0,:,:]
images10 = functions.openData(path + "/1")[0,:,:]
images11 = functions.openData(path + "/2")[0,:,:]
images12 = functions.openData(path + "/3")[0,:,:]

# threshold again
images9 = ndimage.median_filter(np.ma.masked_where(images9 < threshold, images9).filled(0), size= 3)
images10 = ndimage.median_filter(np.ma.masked_where(images10 < threshold, images10).filled(0), size= 3)
images11 = ndimage.median_filter(np.ma.masked_where(images11 < threshold, images11).filled(0), size= 3)
images12 = ndimage.median_filter(np.ma.masked_where(images12 < threshold, images12).filled(0), size= 3)


# Create a figure with two subplots
fig, axs = plt.subplots(3, 4, figsize=(90, 80)) # 60, 100 for big image
# Adjust the space between subplots
fig.subplots_adjust(hspace=0.01, wspace = 0.01)

fontsize = 120

axs[0,0].imshow(images1, cmap = "gray")
axs[0,0].set_title('a)', fontsize = fontsize)
axs[0,0].set_axis_off()

axs[0,1].imshow(images2, cmap = "gray")
#axs[0].set_title('First PDF')
axs[0,1].set_axis_off()

axs[0,2].imshow(images3, cmap = "gray")
#axs[0].set_title('First PDF')
axs[0,2].set_axis_off()

axs[0,3].imshow(images4, cmap = "gray")
#axs[0].set_title('First PDF')
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
#axs[0].set_title('First PDF')
axs[2,1].set_axis_off()

axs[2,2].imshow(images11, cmap = "gray")
#axs[0].set_title('First PDF')
axs[2,2].set_axis_off()

axs[2,3].imshow(images12, cmap = "gray")
#axs[0].set_title('First PDF')
axs[2,3].set_axis_off()

# colorbar
for row in range(0, 3):
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



os.chdir("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots")
#plt.tight_layout()
plt.savefig("ScenePredictionsAletschNoTransfer.pdf", dpi = 300)
#plt.show()
