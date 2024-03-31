################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ if use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_batch_norm=use_batch_norm
        self.logsisticRegression = False

        if (len(self.n_hidden) == 0):
            self.logsisticRegression = True



        self.Network_layers = nn.ModuleList()
        #update for the first layers
        in_features = n_inputs

        if( self.logsisticRegression == True):
            linear_layer = nn.Linear(in_features, n_classes)
            nn.init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            self.Network_layers.append(linear_layer)


        else:
            # Add hidden layers
            for hidden_units in n_hidden:
                linear_layer = nn.Linear(in_features, hidden_units)
                nn.init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
                self.Network_layers.append(linear_layer)

                if use_batch_norm:
                    self.Network_layers.append(nn.BatchNorm1d(hidden_units))

                self.Network_layers.append(nn.ELU())
                in_features = hidden_units

            # Add Last layer
            last_layer = nn.Linear(in_features, n_classes)
            nn.init.kaiming_uniform_(last_layer.weight, nonlinearity='relu')
            self.Network_layers.append(last_layer)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """


        #######################
        # PUT YOUR CODE HERE  #
        #######################
        layer_Input = x
        for layer in self.Network_layers:
            y = layer(layer_Input)
            layer_Input = y
        out = y
        return out
        #######################
        # END OF YOUR CODE    #
        #######################



    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
