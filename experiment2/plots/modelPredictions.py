import os
import pickle
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np

# Load the two PDF files as images
images1 = convert_from_path('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/predictions/LSTMAttentionSmall/5018/7.pdf')
images2 = convert_from_path('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/predictions/LSTMAttentionSmall/13615/7.pdf')
images3 = convert_from_path('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment1/predictions/LSTMAttentionSmall/16484/7.pdf')

# Convert the first page of each PDF file to numpy arrays
images1 = np.asarray(images1[0])
images2 = np.asarray(images2[0])
images3 = np.asarray(images3[0])

# Create a figure with two subplots
fig, axs = plt.subplots(3, 1, figsize=(30, 20))
fig.subplots_adjust(hspace=0.001, wspace = 0.001)

axs[0].imshow(images1)
#axs[0].set_title('First PDF')
axs[0].set_axis_off()


axs[1].imshow(images2)
#axs[1].set_title('Second PDF')
axs[1].set_axis_off()

axs[2].imshow(images3)
#axs[1].set_title('Second PDF')
axs[2].set_axis_off()



# Add some padding between the subplots
#fig.subplots_adjust(hspace=0.5)

# Show the plot
plt.tight_layout()
plt.savefig("patchPredictions.pdf", dpi = 1000)
plt.show()

