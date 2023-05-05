import functions
import numpy as np
import pickle

d = functions.loadData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/patched",
                   ["parbatiPatched"])
                       #["2014", "2015"])



d = functions.getTrainTest(d, 4,  0, 0, stationary = False)
#print(d[1])
#print(len(d))

