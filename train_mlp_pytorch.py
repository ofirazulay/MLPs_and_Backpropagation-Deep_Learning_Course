################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt




def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        accuracy

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()  # Set model to eval mode
    true_preds, num_preds = 0., 0.
    with torch.no_grad():
        for inputs, labels in data_loader:
            x_inputs_Val = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])  # Flat The inputs 3D Matrix to matrix of 2D (row for each data sample)
            Model_Output_Validation = model.forward(x_inputs_Val)

            _, pred_labels = torch.max(Model_Output_Validation.data, 1)
            true_preds += (pred_labels == labels).sum().item()
            num_preds += labels.size(0)

        accuracy = true_preds / num_preds
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    n_inputs = 3072
    n_classes = 10
    if(lr==1 or lr==10 or lr==100):
        use_batch_norm = True
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes,use_batch_norm=use_batch_norm)
    model.to(device)

    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    Losses_all_Batchs_In_epochs= []
    val_accuracies = []
    losses_per_epochs = []
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        print(f"epoch Number: {epoch+1}")
        Loss_in_Batch=[]
        batch_count = 0
        total_loss_in_epoch = 0.0

        for inputs, labels in tqdm(cifar10_loader['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])  # Flat The inputs 3D Matrix to matrix of 2D (row for each data sample)
            preds = model(inputs)
            Loss_Value = loss_module(preds, labels)

            Loss_in_Batch.append(Loss_Value.item())
            Loss_Value.backward()
            optimizer.step()

            total_loss_in_epoch += Loss_Value.item()
            batch_count += 1

        loss_per_epoch=total_loss_in_epoch/batch_count
        losses_per_epochs.append(loss_per_epoch)

        Losses_all_Batchs_In_epochs.append(Loss_in_Batch)


        validation_accuracy = evaluate_model(model, cifar10_loader['validation'], num_classes=10)
        val_accuracies.append(validation_accuracy)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            # Use deepcopy to create a new copy of the model (if necessary)
            best_model = deepcopy(model)



    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'], num_classes=10)


    # TODO: Add any information you might want to save for plotting
    logging_info = {
        'Losses_all_Batchs_In_epochs_Array': Losses_all_Batchs_In_epochs,
        'losses_per_epochs_Array': losses_per_epochs,
        'best_accuracy_Value': best_accuracy
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
#--------------------------------------------------------4 PyTorch MLP-------------------------------
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    _, val_accuracies, test_accuracy, logging_info = train(**kwargs)  # Call the train function and assign logging_info to a variable
    print(f"validation accuracies from epoch 1 to epoch 10 is: ")
    print(val_accuracies)
    print(f"Validation best accuracy is :  {logging_info['best_accuracy_Value']} ")
    print(f"Test Accuracy With Best Model: {test_accuracy}")

    Losses_all_Batchs_In_epochs_input_Function = (logging_info['Losses_all_Batchs_In_epochs_Array'])

    # plot for lr=0.1 (loss value for each batch in each epoch)
    def plot_Losses_all_Batchs_In_epochs(Losses_all_Batchs_In_epochs):
        loss_values = np.array(Losses_all_Batchs_In_epochs)
        loss_flattened = loss_values.flatten()
        num_epochs = len(loss_values)
        epochs = np.arange(1, num_epochs + 1)
        fig, ax = plt.subplots()
        ax.plot(loss_flattened, 'orchid')
        ax.set_xlabel('Mini-Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss per Mini-Batch - pytorch-MLP')
        batch_per_epoch = loss_values.shape[1]
        for epoch in range(1, num_epochs):
            ax.axvline(x=epoch * batch_per_epoch, color='lightgray', linestyle='--')
        batch_ticks = np.arange(batch_per_epoch // 2, batch_per_epoch * num_epochs, batch_per_epoch)
        ax.set_xticks(batch_ticks)
        ax.set_xticklabels([f'{batch_num}\nEpoch {epoch_num}' for batch_num, epoch_num in zip(batch_ticks, epochs)])
        last_loss_values = loss_values[:, -1]
        for epoch, loss in zip(epochs, last_loss_values):
            ax.annotate(f'{loss:.4f}', xy=(epoch * batch_per_epoch, loss), xytext=(epoch * batch_per_epoch, loss),
                        ha='center', va='bottom', fontsize=6)
        plt.subplots_adjust(bottom=0.2)
        for label in ax.get_xticklabels():
            label.set_fontsize(6)
        plt.show()
    #run plot of loss function
    plot_Losses_all_Batchs_In_epochs(Losses_all_Batchs_In_epochs_input_Function)


#----------------------------------------------4.1 Learning rate-----------------------

    learning_rates = np.logspace(-6, 2, num=9)  # Generate logarithmically spaced learning rates
    loss_curves_lr = []
    loss_curves_Big_lr=[]
    best_accuracies_lr=[]

    for lr in learning_rates:
        _, val_accuracies, test_accuracy, logging_info = train([128], lr, False, 128, 10, 42, 'data/')
        print(f"validation accuracies from epoch 1 to epoch 10 of learning_rate: {lr} is: ")
        print(val_accuracies)
        print(f"Validation best accuracy is :  {logging_info['best_accuracy_Value']} ")
        print(f"Test Accuracy With Best Model of learning_rate: {lr} is: {test_accuracy}")


        if lr not in [10, 100]:
            loss_curves_lr.append(logging_info['losses_per_epochs_Array'])  # Append logging_info to the list
        else:
            loss_curves_Big_lr.append(logging_info['losses_per_epochs_Array'])

        best_accuracies_lr.append(logging_info['best_accuracy_Value'])



#-----------------------------------4.1 plots functions (different learning_rates)
    def plot_best_accuracies_lr(best_accuracies_lr):
        learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        bar_width = 0.8
        x_pos = np.arange(len(learning_rates))
        plt.figure(figsize=(10, 8))
        plt.bar(x_pos, best_accuracies_lr, width=bar_width, color='palevioletred')
        plt.xlabel('Learning Rate', fontsize=12, fontweight='bold')
        plt.ylabel('Best Accuracy', fontsize=12, fontweight='bold')
        plt.title('Best Accuracy vs Learning Rate', fontsize=20, fontweight='bold')
        plt.xticks(x_pos, learning_rates, rotation='vertical')
        for i, v in enumerate(best_accuracies_lr):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.show()

    def plot_loss_curves_for_all_regular_lr(loss_curves_lr):
        learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        epochs = range(1, 11)
        colors = ['pink', 'g', 'purple', 'c', 'orange', 'olive', 'b']
        for i, lr_curve in enumerate(loss_curves_lr):
            plt.plot(epochs, lr_curve, label=f'Learning Rate {learning_rates[i]}', color=colors[i])
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.title('Loss Curves for Different Learning Rates')
        plt.xticks(epochs)
        plt.legend()
        plt.show()

    def plot_loss_curves_for_Big_lr1(loss_curves_Big_lr):
        learning_rates = [10, 100]
        epochs = range(1, 11)
        fig, axs = plt.subplots(len(learning_rates), 1, figsize=(8, 6), sharex=True)
        for i, lr in enumerate(learning_rates):
            axs[i].plot(epochs, loss_curves_Big_lr[i])
            axs[i].set_ylabel('Loss')
            axs[i].set_title('Learning Rate=' + str(lr))
        axs[-1].set_xlabel('Epochs')
        plt.tight_layout()
        plt.show()


    #run plot functions (4.1 plots functions (different learning_rates))
    plot_best_accuracies_lr(best_accuracies_lr)
    plot_loss_curves_for_all_regular_lr(loss_curves_lr)
    plot_loss_curves_for_Big_lr1(loss_curves_Big_lr)













    # Feel free to add any additional functions, such as plotting of the loss curve here
