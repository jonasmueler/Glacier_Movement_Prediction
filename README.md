# Glacier Movement Prediction with Deep Learning Models and Satellite Data

## Data Availability
The data used for all three experiments can be accessed in the /mnt/qb/work/ludwig/nludwig/Data/glacier_movement_deep_learning directory of the slurm cluster for compute jobs of the university of tuebingen (https://gitlab.mlcloud.uni-tuebingen.de/doku/public/-/wikis/SLURM). For access please contact jonas3.mueller@student.uni-tuebingen.de or nicole.ludwig@uni-tuebingen.de

The /experiment1 folder contains the /images and /targets folders which contain all data input sequences (images) and target sequences. The dataloader of the first experiment lists all elements in these folders and divides into train/validate and test (80/20 split) for training. The scene folder contains all unpatched scene files, while the dates folder contains the corresponding dates. As the dataloader takes the last 20 % of the data in /images and /targets as test data, the sequence of the last 8 scenes in the /scenes folder just contain testset data and are therefore used in the work for testing of full scene predictions. 

In the /experiment2 folder the scenes were directly divided into train (n = 19) and testscenes (n = 8) because of the experimental design given. Therefore only the scenes not used for the generalization tests are patched into sequences in the /images and /targets folders.

In the /experiment3 folder train data of both experiments was mixed in the /images and /targets folders, while the testdata of the parvati glacier was given in the /parvatiTest folder in patched format and the testdata of the altesch glaciers was given in /scenesAletsch in scene format, based on the natur of the created data for experiment 1 and 2.

## Experiment 1 
/experiment1

### Data Acquisition and Preprocessing

In order to download the necessary satellite scenes from the planetary computer interface the function the dataAPI.py script in the preProcessing directory can be used. The API function takes a tuple of four coordinates as a bounding box in longitude and lattitude coordinates (can be created for example extracted with the ipyleaflet tool for python), a timerange as a string representation, the maximal allowed cloud coverage in percent, the maximal allowed ratio of missing pixels (in [0,1]), the year (string), the glacierName (string) and a boolean value indicating if plots should be created for data validtaion, as input. With these arguments landsat-8 spectral images can be extracted for the whole globe (see dataAPI.py for large scale extraction). Remember to change the path to the location you want to extract the data to. This script will create a folder for the data of the glacier with one pickle file containing the scene data for each year from 2013 until 2021. The createPatches.py script then preprocesses the raw landsat-8 data, therefore applies a kernel in order to clean missing values (Vonica et al. 2021), aligns all scenes with enhanced correlation coefficient maximization to a median filtered mean image, saves these aligned scenes, and divides the scenes into equally sized patches. The output is then saved in one big pickle file. With the getData.py script this file is then used to create the sequences that are used for training and testing. Remember to change the path in the getTrainTest function in the functions.py script in order to specify where all the train data (input, targets) is saved and to change the root directory used in the pathOrigin variable at the beginning of functions.py. Keep in mind that now a file for every sequence is created. 

### Model Training
The models were trained on a high-performance GPU cluster of the university of tuebingen. Therefore a singularity container was used in order to run training scripts. The deeplearning.def script creates a singularity container from a docker image containing a ubuntu 20.04 operating system, ready to be used for model training with pytorch and cuda. The container can be created with the following commands: 

```
singularity build --fakeroot deeplearning.sif <path to deeplearning.def>

```
This container can then be used in order to run scripts, for example without entering the container via:

```
singularity exec --nv --bind <path to repo that should be bounded to container>,`pwd` deeplearning.sif python3 <path to script> --timer_repetitions 10000 --gpu
```
In order to train the models three folders were created /LSTM, /unet, /convLSTM. With the LSTMTrain.py and lstmAttentionTrain.py files in the LSTM folder hyperaparemeters can be specified and the model training can be started (on GPU with Cuda). Note that the scripts use the weights and biases tool, which is a free tool to monitor training progress in real time in a browser application. If weights and biases should be used, then a account has to be created and registered in the used environemnt, if not the wandb argument can be set to false in the trainLoop function. The same procedure can be applied to convLSTMTrain.py in the convLSTM, and train.py in the unet directory. The models are saved in a created subdirectory /models, where also a csv file is stored with the train and validation losses.

The trained models of all experiments can be assessed here: https://drive.google.com/drive/folders/1eyJuXyeawjjeRVgVCfjP_opeBiFG5cEg?usp=sharing

### Testset performance
With the testsetPerformance.py script in the /testing directory the models can be tested on the testset. The glaciers class from the datasetClasses.py script has a bootstrap argument, which enables bootstrap sampling of the testset if set to True, otherwise MSE and MAE scores are calculated on the testset and stored in a file in the /models folder. 

In order to create predictions on the full scenes the inferencePlot.py script in the /plots folder can be used. The model tested has to be uncommented in order for the model and the weights to be loaded. The function takes a scene sequence of 8 scenes from the testset, splits the scenes into patch sequences, always uses the first 4 patches as model input and predicts the last 4 patches. The predictions are then put together again in order to get the full scenes and saved together with the target scene images in the /results folder. 

### Optical flow
With the opticalFlow.py sript in the /plots directory then optical flow vectors were estimated and plotted for model predictions and for target scene masks. 

## Experiment 2 
\experiment2

### Data Acquisition and Preprocessing
The data acquisition is done in the same order described in the first experiment, note that now not all scenes are used, as scenes had to be scanned manually in order to get rid of biased scenes. In order to have a sufficiently large dataset the last 8 scenes of the aligned scenes were used for testing, therefore training patch sequences were only created for the 19 remaining scenes.

### Model Training 
In order to train the pretrained model further on the new glacier data, the lstmAttentionTrain.py in /LSTM script can be used, which loads the old weights and trains the model further based on hyperparameter specifications in the same file. 

### Testset Performance
The testset scores were calculated with the AletschMSE.py sript, while scene predictions were plotted with the aletschGeneralizationTest.py and aletschGeneralizationPlot.py scripts.  

## Experiment 3 
\experiment3

### Data Acquisition 
In this experiment now both datasets from the previous experiments are used together.

### Model Training
Again the lstmAttentionTrain.py in /LSTM script can be used in order to train the model. Note that now the dataloader uses the /aletsch folder as the mixed dataset was put into this directory. 

### Testset Performance
The testset scores were calculated with the AletschMSE.py and the testsetPerformanceLSTMAttention.py sript in /testing. Predictions were plotted with the aletschGeneralizationTest.py, aletschGeneralizationPlot.py and inferencePlot.py scripts in /plots for respective datasets. 

 














