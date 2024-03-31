################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################

"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):

        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logsistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.linear_modules_layers = []
        self.RELU_modules_layers = []
        self.softMex_layer=[]
        self.logsisticRegression = False

        if (len(self.n_hidden) == 0):
            self.logsisticRegression = True
            linear_module = LinearModule(n_inputs, n_classes, input_layer=True)
            self.linear_modules_layers.append(linear_module)
            SoftMax_Module = SoftMaxModule()
            self.softMex_layer.append(SoftMax_Module)


        else:
            num_hidden = len(self.n_hidden)
            for i in range(num_hidden + 1):
                is_first = (i == 0)
                is_last = (i == num_hidden)

                if (is_first):
                    units_current_layer = self.n_hidden[i]
                    linear_module = LinearModule(n_inputs, units_current_layer, input_layer=True)
                    RELU_module=ELUModule()
                    self.RELU_modules_layers.append(RELU_module)

                elif (is_last):
                    units_Previous_layer = self.n_hidden[i - 1]
                    linear_module = LinearModule(units_Previous_layer, n_classes, input_layer=False)
                    SoftMax_Module=SoftMaxModule()
                    self.softMex_layer.append(SoftMax_Module)

                else:
                    units_Previous_layer = self.n_hidden[i - 1]
                    units_current_layer = self.n_hidden[i]
                    linear_module = LinearModule(units_Previous_layer, units_current_layer, input_layer=False)
                    RELU_module = ELUModule()
                    self.RELU_modules_layers.append(RELU_module)

                self.linear_modules_layers.append(linear_module)


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:s
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        if (self.logsisticRegression):
            self.Linear_layer_Current_Output = self.linear_modules_layers[0].forward(x)
            self.SoftMax_Current_Output = self.softMex_layer[0].forward(self.Linear_layer_Current_Output)

        else:
            num_Of_layers = len(self.linear_modules_layers)

            self.Linear_layer_Current_Output=None
            self.RELU_Activation_Current_Output=None
            self.SoftMax_Current_Output=None

            for i in range(num_Of_layers):
                is_first = (i == 0)
                is_last = (i == num_Of_layers-1)

                if (is_first):
                    self.Linear_layer_Current_Output=self.linear_modules_layers[i].forward(x)
                    self.RELU_Activation_Current_Output =self.RELU_modules_layers[i].forward(self.Linear_layer_Current_Output)


                elif (is_last):
                    self.Linear_layer_Current_Output = self.linear_modules_layers[i].forward(self.RELU_Activation_Current_Output)
                    self.SoftMax_Current_Output =self.softMex_layer[0].forward(self.Linear_layer_Current_Output)


                else:
                    self.Linear_layer_Current_Output = self.linear_modules_layers[i].forward(self.RELU_Activation_Current_Output)
                    self.RELU_Activation_Current_Output = self.RELU_modules_layers[i].forward(self.Linear_layer_Current_Output)

        out = self.SoftMax_Current_Output
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss with respec to the network output

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        if (self.logsisticRegression):
           self.SoftMax_Backward_Current_Output = self.softMex_layer[0].backward(dout)
           self.Linear_Backward_layer_Current_Output = self.linear_modules_layers[0].backward(self.SoftMax_Backward_Current_Output)


        else:
            self.SoftMax_Backward_Current_Output = self.softMex_layer[0].backward(dout)

            num_Of_layers = len(self.linear_modules_layers)
            for i in range(num_Of_layers - 1, -1, -1):
                if i == num_Of_layers - 1:
                    self.Linear_Backward_layer_Current_Output=self.linear_modules_layers[i].backward(self.SoftMax_Backward_Current_Output)

                else:
                    self.RELU_Backward_Current_Output = self.RELU_modules_layers[i].backward(self.Linear_Backward_layer_Current_Output) ### self.linear_Backward_Current_Output - self.Linear_Backward_layer_Current_Output
                    self.linear_Backward_Current_Output = self.linear_modules_layers[i].backward(self.RELU_Backward_Current_Output)

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################
