################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2023b
#
################################################################################

"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np
from copy import deepcopy

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #######################

        std = np.sqrt(2 / in_features)
        self.Weights = np.random.normal(0, std, (out_features, in_features))
        self.biases = np.full(out_features, 0)
        W_gradient = np.full((out_features,in_features), 0)
        b_gradient = np.full(out_features, 0)
        self.grads = {'weight': W_gradient, 'bias': b_gradient}

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.Linear_x=x

        self.f_Linear = x.dot(self.Weights.T)+self.biases
        out=self.f_Linear

        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = (dout.T).dot(self.Linear_x)

        allOne_Vector = np.full(dout.shape[0], 1)
        self.grads['bias'] = allOne_Vector.dot(dout)

        dx = dout.dot(self.Weights)

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################



class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.RELU_x=x

        self.h_RELU=np.maximum(0, x)
        out=self.h_RELU
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        U_Output=np.where(self.RELU_x > 0, 1, 0)
        dx=dout*U_Output
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.Softmax_x = x

        X = x - np.max(x, axis=1, keepdims=True)
        exp_X = np.exp(X)
        sum_exp_X = np.sum(exp_X, axis=1, keepdims=True)
        self.s_Softmax = exp_X / sum_exp_X

        out=self.s_Softmax
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        C=self.s_Softmax.shape[1]
        AllOneMatrix = np.full((C, C), 1)

        dx=(dout*self.s_Softmax)-((dout*self.s_Softmax).dot(AllOneMatrix))*self.s_Softmax

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.S = x.shape[0]
        L_i=-(np.sum(y*np.log(x),axis=1))
        L_AVG=(1/self.S)*np.sum(L_i)
        out=L_AVG
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        coef=-1/(self.S)
        dx=coef*(y/x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx