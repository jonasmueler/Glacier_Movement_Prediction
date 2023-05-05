# Glacier Movement Prediction with Deep Learning Models and Satellite Data

## Experiment 1 

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

