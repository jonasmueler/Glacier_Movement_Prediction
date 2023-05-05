# Glacier Movement Prediction with Deep Learning Models and Satellite Data

## Experiment 1 
/experiment1

### Data Acquisition and Preprocessing

In order to download the necessary satellite scenes from the planetary computer interface the function the dataAPI.py script in the preProcessing directory can be used. Remember to change the path to the location you want to extract the data to. This script will create a folder for the data of the glacier with one pickle file containing the scene data for each year from 2013 until 2021. The createPatches.py script then preprocesses the raw landsat-8 data, therefore applies a kernel in order to clean missing values (Vonica et al. 2021), aligns all scenes with enhanced correlation coefficient maximization to a median filtered mean image, and divides the scenes into equally sized patches. The output is then saved in one big pickle file. With the getData.py script this file is then used to create the sequences that are used for training and testing. Remember to change the path in the getTrainTest function in the functions.py script in order to specify where all the train data (input, targets) is saved. Keep in mind that now a file for every sequence is created. 

### Model Training
The models were trained on a high-performance GPU cluster. Therefore a singularity container was used in order to run training scripts. The deeplearning.def script creates a singularity container from a docker image containing a ubuntu 20.04 operating system, ready to be used for model training with pytorch and Cuda. The container can be created with the following commands: 

```
singularity build --fakeroot deeplearning.sif <path to deeplearning.def>

```
This container can then be used in order to run scripts, for example without entering the container via:

```
singularity exec --nv --bind <path to repo that should be bounded to container>,`pwd` deeplearning.sif python3 <path to script> --timer_repetitions 10000 --gpu
```
In order to train the models three folders were created /LSTM, /unet, /convLSTM. With the LSTMTrain.py and lstmAttentionTrain.py files in the LSTM folder hyperaparemeters can be specified and the model training can be started (on GPU with Cuda). Note that the scripts use the weights and biases tool, which is a free tool to monitor training progress in real time in a browser application. If weights and biases should be used, then a account has to be created and registered in the used environemnt, if not the wandb argument can be set to false in the trainLoop function. The same procedure can be applied to convLSTMTrain.py in the convLSTM, and train.py in the unet directory. The models are saved in a created subdirectory /models, where also a csv file is stored with the train and validation losses.

### Testset performance
With the testsetPerformance.py script in the /testing directory the models can be tested on the testset. The glaciers class from the datasetClasses.py script has a bootstrap argument, which enables bootstrap sampling of the testset if set to True, otherwise MSE and MAE scores are calculated on the testset and stored in a file in the /models folder. 

In order to create predictions on the full scenes the inferencePlot.py script in the /plots folder can be used. The model tested has to be uncommented in order for the model and the weights to be loaded. The function takes a scene sequence of 8 scenes from the testset, splits the scenes into patch sequences, always uses the first 4 pacthes as model input and predicts the last 4 patches. The predictions are then put together again in order to get the full scene and saved together with the target scene images in the /results folder. 











