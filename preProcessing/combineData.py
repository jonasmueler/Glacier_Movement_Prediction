import functions
import pickle
import os

"""
## get train data
# aletsch images
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn/patched"
images = os.listdir(os.path.join(path, "images"))
pathsImg = [os.path.join(os.path.join(path, "images"), item) for item in images if os.path.isfile(os.path.join(os.path.join(path, "images"), item))]

targets = os.listdir(os.path.join(path, "targets"))
pathsTargets = [os.path.join(os.path.join(path, "targets"), item) for item in targets if os.path.isfile(os.path.join(os.path.join(path, "targets"), item))]

counter = 0
for i in range(len(pathsImg)):
    # images
    tensor = functions.openData(pathsImg[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/trainData/images"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    # targets
    tensor = functions.openData(pathsTargets[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/trainData/targets"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    counter += 1

# helheim
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim/patched"
images = os.listdir(os.path.join(path, "images"))
pathsImg = [os.path.join(os.path.join(path, "images"), item) for item in images if os.path.isfile(os.path.join(os.path.join(path, "images"), item))]

targets = os.listdir(os.path.join(path, "targets"))
pathsTargets = [os.path.join(os.path.join(path, "targets"), item) for item in targets if os.path.isfile(os.path.join(os.path.join(path, "targets"), item))]


for i in range(len(pathsImg)):
    # images
    tensor = functions.openData(pathsImg[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/trainData/images"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    # targets
    tensor = functions.openData(pathsTargets[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/trainData/targets"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    counter += 1

# helheim
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jakobshavn/patched"
images = os.listdir(os.path.join(path, "images"))
pathsImg = [os.path.join(os.path.join(path, "images"), item) for item in images if os.path.isfile(os.path.join(os.path.join(path, "images"), item))]

targets = os.listdir(os.path.join(path, "targets"))
pathsTargets = [os.path.join(os.path.join(path, "targets"), item) for item in targets if os.path.isfile(os.path.join(os.path.join(path, "targets"), item))]


for i in range(len(pathsImg)):
    # images
    tensor = functions.openData(pathsImg[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/trainData/images"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    # targets
    tensor = functions.openData(pathsTargets[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/trainData/targets"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    counter += 1

"""
### get validation data
# aletsch images
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn/patched"
images = os.listdir(os.path.join(path, "images", "validation"))
pathsImg = [os.path.join(os.path.join(path, "images", "validation"), item) for item in images if os.path.isfile(os.path.join(os.path.join(path, "images", "validation"), item))]

targets = os.listdir(os.path.join(path, "targets", "validation"))
pathsTargets = [os.path.join(os.path.join(path, "targets", "validation"), item) for item in targets if os.path.isfile(os.path.join(os.path.join(path, "targets", "validation"), item))]

counter = 0
for i in range(len(pathsImg)):
    # images
    tensor = functions.openData(pathsImg[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/valData/images"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    # targets
    tensor = functions.openData(pathsTargets[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/valData/targets"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    counter += 1

# helheim
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim/patched"
images = os.listdir(os.path.join(path, "images", "validation"))
pathsImg = [os.path.join(os.path.join(path, "images", "validation"), item) for item in images if os.path.isfile(os.path.join(os.path.join(path, "images", "validation"), item))]

targets = os.listdir(os.path.join(path, "targets", "validation"))
pathsTargets = [os.path.join(os.path.join(path, "targets", "validation"), item) for item in targets if os.path.isfile(os.path.join(os.path.join(path, "targets", "validation"), item))]

for i in range(len(pathsImg)):
    # images
    tensor = functions.openData(pathsImg[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/valData/images"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    # targets
    tensor = functions.openData(pathsTargets[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/valData/targets"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    counter += 1

# helheim
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jakobshavn/patched"
images = os.listdir(os.path.join(path, "images", "validation"))
pathsImg = [os.path.join(os.path.join(path, "images", "validation"), item) for item in images if os.path.isfile(os.path.join(os.path.join(path, "images", "validation"), item))]

targets = os.listdir(os.path.join(path, "targets", "validation"))
pathsTargets = [os.path.join(os.path.join(path, "targets", "validation"), item) for item in targets if os.path.isfile(os.path.join(os.path.join(path, "targets", "validation"), item))]

for i in range(len(pathsImg)):
    # images
    tensor = functions.openData(pathsImg[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/valData/images"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    # targets
    tensor = functions.openData(pathsTargets[i])
    # save into folder
    currentPath = os.getcwd()
    outputPath = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/valData/targets"
    os.chdir(outputPath)
    with open(str(counter), "wb") as fp:  # Pickling
        pickle.dump(tensor, fp)
    os.chdir(currentPath)

    counter += 1