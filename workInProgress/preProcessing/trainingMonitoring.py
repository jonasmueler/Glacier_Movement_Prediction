import pandas as pd
import matplotlib.pyplot as plt

TransformerSmall = pd.read_csv("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/models/TransformerSmall.csv")
TransformerMiddle = pd.read_csv("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/models/TransformerMiddle.csv")
TransformerBig = pd.read_csv("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/models/TransformerBig.csv")
LSTM = pd.read_csv("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/models/LSTM.csv")
TransformerSmall = TransformerSmall.iloc[1:25000, :]
TransformerMiddle = TransformerMiddle.iloc[1:25000, :]
TransformerBig = TransformerBig.iloc[1:25000, :]

# plot; averaged
cmap = plt.get_cmap("summer")

plt.plot(TransformerSmall.groupby("epoch")["Validation Loss"].mean(),
         color=cmap(0.2),
         label ="Transformer (2 Encoding/Decoding Layers, 4 Attention Heads)")
plt.plot(TransformerMiddle.groupby("epoch")["Validation Loss"].mean(),
         color=cmap(0.5),
         label = "Transformer (4 Encoding/Decoding Layers, 8 Attention Heads)")
plt.plot(TransformerBig.groupby("epoch")["Validation Loss"].mean(),
         color=cmap(0.8),
         label = "Transformer (6 Encoding/Decoding Layers, 10 Attention Heads)")

plt.plot(TransformerSmall.groupby("epoch")["Train Loss"].mean(),
         color=cmap(0.2),
        linestyle='dashed')
plt.plot(TransformerMiddle.groupby("epoch")["Train Loss"].mean(),
         color=cmap(0.5),
        linestyle='dashed')
plt.plot(TransformerBig.groupby("epoch")["Train Loss"].mean(),
         color=cmap(0.8),
        linestyle='dashed')

plt.legend(fontsize = 8)
#plt.savefig()
plt.show()


# plot; not averaged
cmap = plt.get_cmap("summer")

plt.plot(TransformerSmall["Validation Loss"],
         color=cmap(0.2),
         label ="Transformer (2 Encoding/Decoding Layers, 4 Attention Heads)")
plt.plot(TransformerMiddle["Validation Loss"],
         color=cmap(0.5),
         label = "Transformer (4 Encoding/Decoding Layers, 8 Attention Heads)")
plt.plot(TransformerBig["Validation Loss"],
         color=cmap(0.8),
         label = "Transformer (6 Encoding/Decoding Layers, 10 Attention Heads)")
"""
plt.plot(TransformerSmall["Train Loss"],
         color=cmap(0.2),
        linestyle='dashed')
plt.plot(TransformerMiddle["Train Loss"],
         color=cmap(0.5),
        linestyle='dashed')
plt.plot(TransformerBig["Train Loss"],
         color=cmap(0.8),
        linestyle='dashed')
"""
plt.legend(fontsize = 8)
#plt.savefig()
plt.show()