# Virulence Prediction Model for Host Pathogen Protein-Protein Interactions
A project for NTU MSc Artificial Intelligence course. 

In this project, we design and implement a model that is capable predicting the virulence class of protein interactions between mouse and virus proteins.
We also curate a dataset for training and evaluating our model.

## Environment Setup
Our project is built on a Python 3.9.18 environement, with WSL2 as the system OS.

The Anaconda and PIP commands we used to setup our environment can be found in `EnvSetupCommands.txt`

## Adding the Dataset to the Data Folder
Before training the model, the dataset files must first be in the `data/` folder.<br>
(Data files To be update)

The program will perform train-validation-test split on this dataset at a ratio of 70-20-10.

## Training the Model
To start training the model run:
```
python train.py
```
The program will train the model for a total of 200 epochs.

To resume training from a specific checkpoint, run the following command.
```
python train.py -cpf checkpoints/<name_of_your_checkpoint_folder>
```
Example: `python train.py -cpf checkpoints/20240113_024123_epoch100`

## About the Checkpoint Folders
As mentioned above, checkpoint folders are generated inside the `checkpoints/` folder periodically. 
Specifically, this happens once every 10 epochs.

The folders are labeled by timestamp and number of completed epochs. 
For example `checkpoints/20240113_024123_epoch100/` folder is generated at 13 Jan 2024 at 02:41:23 after Epoch 100 in the training process.

Additionally, checkpoints folders generated on training completion are append with a `_fin` suffix.

The folder will contain:
* `checkpoint.pth`: Checkpoint file that stores a dictionary with the following information.
  * Number of Completed Epochs (`dictionary key: epoch`)
  * Model's State Dictionary  (`dictionary key: model_state`)
  * Optimizer's State Dictionary (`dictionary key: optim_state`)
* `last_validationset_classification_report.txt`: Classification report for the model's performance validation set on last completed epoch
* `metric_results.csv`: A running record of per-epoch results. For each epoch, the following information are available.
  * Average Train Loss
  * Per-Class F1 Score on Training Set
  * Overall Accuracy on Training Set
  * Average Validation Loss
  * Per-Class F1 Score on Validation Set
  * Overall Accuracy on Validation Set
* `test_results.txt`: Only available if you have used the checkpoint to evaluate the trained model via the `test.py` script. This file contains a record of the model's performance on the test set.

## Evaluating on the Test Set
To evaluate a trained model against the test set, run the following command.
```
python test.py -cpf checkpoints/<name_of_your_checkpoint_folder>
```
(Example)<br>
You have completed an entire round of training and the program generated the final checkpoint folder `20240113_041818_epoch200_fin/` within the `checkpoints/` folder (i.e. full path to checkpoint folder is `checkpoints/20240113_041818_epoch200_fin/`).

The command to evaluate that trained model will be `python test.py -cpf checkpoints/20240113_041818_epoch200_fin`<br><br>

After evaluation on test set is completed, the results seen on the console will also be printed to a `test_results.txt` file. This file can be found in the checkpoint folder that was indicated when running the above command.
