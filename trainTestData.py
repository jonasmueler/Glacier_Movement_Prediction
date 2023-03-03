import functions
import numpy as np
import pickle

d = functions.loadData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim/patched",
                   ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"])
#d = functions.loadData("D:/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/repo/Helheim",
#            ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"])
#print(len(d))


d = functions.getTrainTest(d, 5,  [7, 8, 9], 9)
#print(d[1])
print(len(d))

