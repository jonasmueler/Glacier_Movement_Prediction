#Glacier Movement Prediction with Deep Learning Models and Satellite Data

## Experiment 1 

### Data Acquisition and Preprocessing

In order to download the necessary satellite scenes from the planetary computer interface the function the dataAPI.py script in the preProcessing directory can be used. Remember to change the path to the location you want to extract the data to. This script will create a folder for the data of the glacier with one pickle file containing the scene data for each year from 2013 until 2021. The createPatches.py script then preprocesses the raw landsat-8 data, therefore applies a kernel in order to clean missing values (Vonica et al. 2021), aligns all scenes with enhanced correlation coefficient maximization to a median filtered mean image, and divides the scenes into equally sized patches. The output is then saved in one big pickle file. With the getData.py script this file is then used to create the sequences that are used for training and testing. Remember to change the path in the getTrainTest function in the functions.py script in order to specify where all the train data (input, targets) is saved. Keep in mind that now a file for every sequence is created. 
