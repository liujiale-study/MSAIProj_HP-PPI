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

The dataset is used to construct a graph to be subsampled from and processed by our GNN.
Our GNN is designed to do edge prediction, and before training or evaluation, the program will perform train-validation-test split on the graph edges at a ratio of 70:20:10.
Training set edges are additionally split into message passing and supervision edges at the ratio of 70:30.

## Training the Model
The training script will train the model for 150 epochs.
During this time, the program will also record the best fit model based on loss on validation set.
Periodically after a set number of epochs, both the current model (i.e. the model that is trained thus far) and the best fit model will be checkpointed, generating a checkpoint folder within the `checkpoints/` folder.

To start training the model using the default GNN operator (RGGCN operator), simply run:
```
python train.py
```

To resume training from a specific checkpoint, run the following command.
```
python train.py -cpf checkpoints/<name_of_your_checkpoint_folder>
```
Example: `python train.py -cpf checkpoints/20240113_024123_epoch20_RGGCN`

To train the model with a specific GNN operator, run the following command
```
python train.py -g <ID_of_gnn_operator>
```
Example: `python train.py -g 1` to train the model while it uses the GAT operator.

The operators of the following GNN are implemented in this project. Their corresponding IDs are also as follows, alongside any key model parameters that do not use their default values.
* [Residual Gated Graph Convolutional Network (RGGCN)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ResGatedGraphConv.html): ID `0` (default operator)
* [Graph Attentional Network (GAT)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html): ID `1`
  * `heads=4`, `concat=False`
* [Graph Transformer (GTR)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html): ID `2`
  * `heads=5`, `concat=False`
* [Graph Isomorphism with Edge Features (GINE)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html): ID `3`
  * `eps=0.0`, `train_eps=True`

## About the Checkpoint Folders
As mentioned above, checkpoint folders are generated inside the `checkpoints/` folder periodically. 
Specifically, this happens once every 10 epochs.

The folders are labeled by timestamp, number of completed epochs and the type of GNN operator used. 
For example, `checkpoints/20240113_024123_epoch20_RGGCN/` folder is generated at 13 Jan 2024 at 02:41:23 after Epoch 20 in the training process, with the model using the RGGCN operator.

Additionally, checkpoints folders generated on training completion are append with a `_fin` suffix.

The folder will contain:
* `checkpoint.pth`: Checkpoint file that stores a dictionary with the following information.
  * Number of Completed Epochs (`dictionary key: epoch`)
  * Current Model's State Dictionary  (`dictionary key: model_state`)
  * ID of the type of GNN operator used by the model (`dictionary key: model_gnn_op_type`)
  * Optimizer's State Dictionary (`dictionary key: optim_state`)
  * State Dictionary of Best Fit Model  (`dictionary key: bestmodel_state`)
  * Epochs elapsed by Best Fit Model  (`dictionary key: bestmodel_epoch`)
* `last_validationset_classification_report.txt`: [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) for the model's performance on the validation set for the last completed epoch
* `metric_results.csv`: A running record of per-epoch results. For each epoch, the following information are available.
  * Mean Train Loss
  * Per-Class F1 Score on Training Set
  * Overall Accuracy on Training Set
  * Mean Validation Loss
  * Per-Class F1 Score on Validation Set
  * Overall Accuracy on Validation Set
* `test_results_currmodel.txt`: Only available if you have evaluated the checkpointed models via the `test.py` script. This file contains a record of the current model's performance on the test set with the following information.
  * Number of Training Epochs the Model Underwent
  * Average Test Loss
  * Matthews Correlation Coefficient
  * [Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
* `test_results_bestmodel.txt`: Only available if you have evaluated the checkpointed models via the `test.py` script. This file contains a record of the best fit model's performance on the test set.
  * Has the same type of information as `test_results_currmodel.txt`, but are based on the best fit model's performance instead.

## Evaluating on the Test Set
To evaluate checkpointed models against the test set, run the following command.
```
python test.py -cpf checkpoints/<name_of_your_checkpoint_folder>
```
(Example)<br>
You have completed a full round of training which lasted 150 epochs, and the program generated the final checkpoint folder `20240113_041818_epoch150_RGGCN_fin/` within the `checkpoints/` folder (i.e. path to checkpoint folder from the repo's top directory is `checkpoints/20240113_041818_epoch150_RGGCN_fin/`).

To evaluate both the model at epoch 150 and the best fit model, run the command `python test.py -cpf checkpoints/20240113_041818_epoch100_RGGCN_fin`<br><br>

After evaluation on test set is completed, the results will be printed to console, as well as the files `test_results_currmodel.txt` and `test_results_bestmodel.txt`.
These files can be found in the checkpoint folder that was indicated when running the above command.

