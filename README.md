# Virulence Prediction Model for Host-IAV Protein-Protein Interactions
A project for NTU MSc Artificial Intelligence course. 

In this project, we design and implement a model that utilises Graph Neural Networks (GNN) to predict the virulence class of protein-protein interactions between mouse and IAV proteins.
We also create a dataset for training and evaluating our model.

## Environment Setup
Our project is built on a Python 3.9.18 environement, with WSL2 as the system OS.

The Anaconda and PIP commands we used to setup our environment can be found in `EnvSetupCommands.txt`

## Adding the Dataset to the Data Folder
Before training the model, the dataset files must first be in the `data/` folder.

Download our zipped dataset file [[data.7z](https://drive.google.com/file/d/14-blNX-A8Y_cuFcGMr7RhixJRrCdZzgZ/view?usp=sharing)], unzip it and put its contents in the `data/` folder. <br>
The following dataset files must be in the `data/` folder for the program to work.
* `mouse_proteins.csv`
* `pp_interactions.csv`
* `virus_proteins.csv`

Before training or evaluation, the program will perform train-validation-test split on this dataset at a ratio of 70-20-10.

## Training the Model
The training script will train the model for 100 epochs.

To start training the model using the default GNN operator (residual gated graph convolutional operator), simply run:
```
python train.py
```

To resume training from a specific checkpoint, run the following command.
```
python train.py -cpf checkpoints/<name_of_your_checkpoint_folder>
```
Example: `python train.py -cpf checkpoints/20240113_024123_epoch20_ResGatedGraphConv`

To train the model with a specific GNN operator, run the following command
```
python train.py -g <ID_of_gnn_operator>
```
Example: `python train.py -g 1` to train the model while it uses the graph attentional operator.

The available GNN operators and their corresponding IDs are as follows.
* [Residual Gated Graph Convolutional Operator (ResGatedGraphConv)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ResGatedGraphConv.html):`0` (default)
* [Graph Attentional Operator (GAT)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html): `1`
* [Graph Transformer Operator (GraphTransformer)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html): `2`
* [Graph Isomorphism Operator with Edge Features (GINE)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html): `3`

## About the Checkpoint Folders
As mentioned above, checkpoint folders are generated inside the `checkpoints/` folder periodically. 
Specifically, this happens once every 10 epochs.

The folders are labeled by timestamp, number of completed epochs and the type of GNN operator used. 
For example, `checkpoints/20240113_024123_epoch20_ResGatedGraphConv/` folder is generated at 13 Jan 2024 at 02:41:23 after Epoch 20 in the training process, with the model using the residual gated graph convolutional operator.

Additionally, checkpoints folders generated on training completion are append with a `_fin` suffix.

The folder will contain:
* `checkpoint.pth`: Checkpoint file that stores a dictionary with the following information.
  * Number of Completed Epochs (`dictionary key: epoch`)
  * Model's State Dictionary  (`dictionary key: model_state`)
  * ID of the type of GNN operator used by the model (`dictionary key: model_gnn_op_type`)
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
You have completed an entire round of training and the program generated the final checkpoint folder `20240113_041818_epoch100_ResGatedGraphConv_fin/` within the `checkpoints/` folder (i.e. full path to checkpoint folder is `checkpoints/20240113_041818_epoch100_ResGatedGraphConv_fin/`).

The command to evaluate that trained model will be `python test.py -cpf checkpoints/20240113_041818_epoch100_ResGatedGraphConv_fin`<br><br>

After evaluation on test set is completed, the results seen on the console will also be printed to a `test_results.txt` file. This file can be found in the checkpoint folder that was indicated when running the above command.
