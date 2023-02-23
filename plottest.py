import functions
import pickle
import matplotlib.pyplot as plt
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/results/transformerScenesSmall/modelPredictions/Helheim/1/targets/0"

with open(path, "rb") as fp:  # Unpickling
    data = pickle.load(fp)

plt.imshow(data[0])
plt.show()

