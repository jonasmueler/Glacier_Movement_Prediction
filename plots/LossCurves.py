import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# create a colormap
cmap = cm.get_cmap('tab10')

# generate a list of colors from the colormap
color = [cmap(i) for i in np.linspace(0, 1, 5)]

# import data
df = pd.read_csv('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots/parvati_training_data/LSTMAttentionSmall.csv')

# get epoch losses

df = df[df['Train Loss'] != 1]
df = df[df['Validation Loss'] != 1]

# get averages over epochs
# define the size of the window
window_size = 1163

# compute the average over every 1600 values of the data
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
plt.plot(averagesVal, label = "Validation Loss SA-LSTM-H", linestyle = "dashed", color = color[0])

# set legend handles and labels

legend_labels = ['Train Loss', 'Validation Loss']

# import data
df = pd.read_csv('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots/parvati_training_data/LSTMEncDec.csv')

# get epoch losses

df = df[df['Train Loss'] != 1]
df = df[df['Validation Loss'] != 1]
#df = df[df['Validation Loss'] != 0]

# get averages over epochs
# define the size of the window
window_size = 1163

# compute the average over every 1600 values of the data
averagesTrain = []
averagesVal = []
for i in range(0, len(df), window_size):
    chunk = df["Train Loss"].iloc[i:i+window_size]
    avg = np.mean(chunk)
    averagesTrain.append(avg)

    chunk = df["Validation Loss"].iloc[i:i + window_size]
    avg = np.mean(chunk)
    averagesVal.append(avg)

plt.plot(averagesTrain, color = color[1], label = "Train Loss LSTM")
plt.plot(averagesVal, linestyle = "dashed", color = color[1], label = "Validation Loss LSTM")
#plt.show()


# import data
df = pd.read_csv('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots/parvati_training_data/Unet.csv')

# get epoch losses

df = df[df['Train Loss'] != 1]
df = df[df['Validation Loss'] != 1]
#df = df[df['Validation Loss'] != 0]

# get averages over epochs
# define the size of the window
window_size = 1150

# compute the average over every 1600 values of the data
averagesTrain = []
averagesVal = []
for i in range(0, len(df), window_size):
    chunk = df["Train Loss"].iloc[i:i+window_size]
    avg = np.mean(chunk)
    averagesTrain.append(avg)

    chunk = df["Validation Loss"].iloc[i:i + window_size]
    avg = np.mean(chunk)
    averagesVal.append(avg)
plt.xlim(left = 0, right = 39)

plt.plot(averagesTrain, color = color[2], label = "Train Loss U-net")
plt.plot(averagesVal, linestyle = "dashed", color = color[2], label = "Validation Loss U-net")


# import data
df = pd.read_csv('/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/code/plots/parvati_training_data/ConvLSTM.csv')

# get epoch losses

df = df[df['Train Loss'] != 1]
df = df[df['Validation Loss'] != 1]
#df = df[df['Validation Loss'] != 0]

# get averages over epochs
# define the size of the window
window_size = 1163

# compute the average over every 1600 values of the data
averagesTrain = []
averagesVal = []
for i in range(0, len(df), window_size):
    chunk = df["Train Loss"].iloc[i:i+window_size]
    avg = np.mean(chunk)
    averagesTrain.append(avg)

    chunk = df["Validation Loss"].iloc[i:i + window_size]
    avg = np.mean(chunk)
    averagesVal.append(avg)
plt.xlim(left = 0, right = 39)
print(len(averagesVal))
plt.plot(averagesTrain, color = color[3], label = "Train Loss ConvLSTM")
plt.plot(averagesVal, linestyle = "dashed", color = color[3], label = "Validation Loss ConvLSTM")
plt.legend(fontsize = 8)
plt.savefig("trainCurves.pdf", dpi = 1000)
plt.show()