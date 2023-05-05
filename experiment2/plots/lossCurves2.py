import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# create a colormap
cmap = cm.get_cmap('viridis')

# generate a list of colors from the colormap
color = [cmap(i) for i in np.linspace(0, 1, 5)]

# import data
df = pd.read_csv('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/experiment2/LSTMAttentionSmallAletschLosses.csv')

# get epoch losses

df = df[df['Train Loss'] != 1]
df = df[df['Validation Loss'] != 1]

print(len(df))


# get averages over epochs
# define the size of the window
window_size = 338

# compute the averages
averagesTrain = []
averagesVal = []
for i in range(0, len(df), window_size):
    chunk = df["Train Loss"].iloc[i:i+window_size]
    avg = np.mean(chunk)
    averagesTrain.append(avg)

    chunk = df["Validation Loss"].iloc[i:i + window_size]
    avg = np.mean(chunk)
    averagesVal.append(avg)

plt.plot(averagesTrain, label = "Train Loss SA-LSTM-H", color = color[0])
plt.plot(averagesVal, label = "Validation Loss SA-LSTM-H", linestyle = "dashed", color = color[2])

# set legend handles and labels

#plt.xlim(left = 0, right = 39)
plt.legend(fontsize = 8)
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.savefig("trainCurvesSecond.pdf", dpi = 1000)
plt.show()

