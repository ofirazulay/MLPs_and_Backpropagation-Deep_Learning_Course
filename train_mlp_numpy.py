################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch
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
    true_preds, num_preds = 0, 0

    for inputs, labels in data_loader:
        x_inputs_Val = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3])#Flat The inputs 3D Matrix to matrix of 2D (row for each data sample)
        Model_Output_Validation = model.forward(x_inputs_Val)

        pred_labels = np.argmax(Model_Output_Validation, axis=1)
        true_preds += np.sum(pred_labels == labels)
        num_preds += labels.shape[0]

    accuracy = true_preds / num_preds
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy




def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)





    #
    # #######################
    # # PUT YOUR CODE HERE  #
    # #######################
    #
    # # TODO: Initialize model and loss module
    #
    n_inputs=3072
    n_classes=10
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=n_classes)
    loss_module = CrossEntropyModule()

    # # TODO: Training loop including validation
    Loss_epoch=[]
    val_accuracies = []
    best_accuracy=0

    for epoch in range(epochs):
        print(f"epoch Number: {epoch+1}")
        Loss_Batch=[]
        for inputs, labels in tqdm(cifar10_loader['train']):
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]) #Flat The inputs 3D Matrix to matrix of 2D (row for each data sample)
            labels= np.eye(n_classes)[labels] #convert labels vector to "One-Hot" Matrix

            MLP_Output = model.forward(inputs)
            Loss_Value = loss_module.forward(MLP_Output, labels)

            Loss_Batch.append(Loss_Value)

            Loss_Grad = loss_module.backward(MLP_Output, labels)
            model.backward(Loss_Grad)
            update_Weights_And_bias(model, lr)

        Loss_epoch.append(Loss_Batch)


        validation_accuracy = evaluate_model(model, cifar10_loader['validation'], num_classes=10)
        val_accuracies.append(validation_accuracy)
    # # TODO: Test best model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_model = deepcopy(model)

    test_accuracy = evaluate_model(best_model, cifar10_loader['test'],num_classes=10)







    # # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'Loss_epoch_Array': Loss_epoch,
        'best_accuracy_Value':best_accuracy
    }

    # #######################
    # # END OF YOUR CODE    #
    # #######################
    #

    return model, val_accuracies, test_accuracy, logging_dict



def update_Weights_And_bias(model,lr):
    for layer in model.linear_modules_layers:
        update_layer(layer,lr)

def update_layer(layer,lr):
    layer.Weights = layer.Weights - lr * layer.grads['weight']
    layer.biases = layer.biases - lr * layer.grads['bias']




if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
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


    _, val_accuracies, test_accuracy, logging_dict = train(**kwargs)  # Call the train function and assign logging_info to a variable
    print(f"validation accuracies from epoch 1 to epoch 10 is: ")
    print(val_accuracies)
    print(f"Validation best accuracy is :  {logging_dict['best_accuracy_Value']} ")
    print(f"Test Accuracy With Best Model: {test_accuracy}")


    Loss_epoch_input_Function = (logging_dict['Loss_epoch_Array'])
def plt_loss_epoch(Loss_epoch_input_Function):
    # Plotting
    loss_values = np.array(Loss_epoch_input_Function)
    loss_flattened = loss_values.flatten()

    num_epochs = len(loss_values)
    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots()
    ax.plot(loss_flattened,'deepskyblue')
    ax.set_xlabel('Mini-Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss per Mini-Batch - numpy-MLP ')

    batch_per_epoch = loss_values.shape[1]
    for epoch in range(1, num_epochs):
        ax.axvline(x=epoch * batch_per_epoch, color='lightgray', linestyle='--')

    batch_ticks = np.arange(batch_per_epoch // 2, batch_per_epoch * num_epochs, batch_per_epoch)
    ax.set_xticks(batch_ticks)
    ax.set_xticklabels([f'{batch_num}\nEpoch {epoch_num}' for batch_num, epoch_num in zip(batch_ticks, epochs)])

    last_loss_values = loss_values[:, -1]
    for epoch, loss in zip(epochs, last_loss_values):
        ax.annotate(f'{loss:.4f}', xy=(epoch * batch_per_epoch, loss), xytext=(epoch * batch_per_epoch, loss),
                    ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(bottom=0.2)

    for label in ax.get_xticklabels():
        label.set_fontsize(6)

    plt.show()

plt_loss_epoch(Loss_epoch_input_Function)