import functions
import matplotlib.pyplot as plt

for i in range(8):
    # load data
    img = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn/monthlyAveragedScenes/images/" + str(18 +i))
    plt.imshow(img)
    plt.show()