# MSAIProj_HP-PPI
Msc Artificial Intelligence Project - Virulence Prediction Model for Host Pathogen Protein-Protein Interactions

# Environment Setup
Our project is built on a Python 3.9.18, with WSL2 as the system OS.

The Anaconda and PIP commands we used to setup our environment can be found in `EnvSetupCommands.txt`

# Adding the Dataset to the Data Folder
Before training the model, the dataset files must first be in the `data/` folder.<br>
(Data files To be update)

The program will perform train-validation-test split on this dataset at a ratio of 70-20-10.

# Training the Model
To start training the model run:
```
python train.py
```
The program will train the model for a total of 200 epochs.
It will also periodically generate timestamped checkpoint folders within the `checkpoints/` folder. 
As the generated folder will contain the training DataLoader object, each generated checkpoint folder can be quite large (500 - 600MB).
Therefore, please ensure you have at least 12GB of free disk space before starting the training process.

To resume training from a specific checkpoint, run the following command.
```
python train.py -cpf checkpoints/<name_of_your_checkpoint_folder>
```
Example: `python train.py -cpf checkpoints/20240113_024123_epoch100`

# About the Checkpoint Folders
As mentioned above, checkpoint folders are generated inside the `checkpoints/` folder periodically. 
Specifically, this happens once every 20 epochs.

The folders are labeled by timestamp and number of completed epochs. 
For example `checkpoints/20240113_024123_epoch100/` folder is generated at 13 Jan 2024 at 02:41:23 after Epoch 100 in the training process.

Additionally, checkpoints folders generated on training completion are append with a `_fin` suffix.

The folder will contain:
* `checkpoint.pth`: Checkpoint file that stores a dictionary with the following information.
  * Number of Completed Epochs (`dictionary key: epoch`)
  * Model's State Dictionary  (`dictionary key: model_state`)
  * Optimizer's State Dictionary (`dictionary key: optim_state`)
* `last_validationset_classification_report.txt`: Classification report for the validation set on last completed epoch
* `metric_results.csv`: A running record of per-epoch results. For each epoch, the following information are available.
  * Average Train Loss
  * Per-Class F1 Score on Training Set
  * Overall Accuracy on Training Set
  * Average Validation Loss
  * Per-Class F1 Score on Validation Set
  * Overall Accuracy on Validation Set
* `test_results.txt`: Only available if you have used the checkpoint to evaluate the trained model via the `test.py` script. The results from the evaluation on the test set will be printed to this file.  

# Evaluating on the Test Set
To evaluate a trained model against the test set, run the following command.
```
python test.py -cpf checkpoints/<name_of_your_checkpoint_folder>
```
Example: You have completed an entire round of training and the program generated the final checkpoint folder `20240113_041818_epoch200_fin/` within the `checkpoints/` folder (i.e. full path to checkpoint folder is `checkpoints/20240113_041818_epoch200_fin/`).

The command to evaluate that trained model will be `python test.py -cpf checkpoints/20240113_041818_epoch200_fin`

After evaluation on test set is completed, the results will also be printed to a `test_results.txt` file. This file can be found in the checkpoint folder that was indicated when running the ``test.py`` script.
