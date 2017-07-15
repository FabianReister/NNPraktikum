
import time

import numpy as np

from util.activation_functions import Activation
#from model.layer import Layer


class LogisticLayer():
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn + 1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1)) - 0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

        self.__counter = 0
        self.__cummulated_gradient = 0

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        # TODO bias?
        input = np.insert(input, input.shape[0], 1)
        self.y_i = input
        self.y_j = self.activation(self.weights.dot(input))
        return self.y_j

    def computeDerivative(self, nextDerivatives):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
       
        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        # FIXME the dimensions of dE_dyj and dyj_dx do not match (bias...)
        dE_dyj = nextDerivatives
        dyj_dx = Activation.sigmoidPrime(self.y_j)
        dx_dw = self.y_i

        dE_dx = dE_dyj * dyj_dx
        dE_dw = dE_dx * dx_dw
        dE_dyi = dE_dx * self.weights

        if self.__counter == 0:
            self.__cummulated_gradient = dE_dw
        else:
            self.__cummulated_gradient += dE_dw
        self.__counter += 1

        return dE_dyi

    def __resetCummulatedGradient(self):
        self.__counter = 0
        self.__cummulated_gradient = 0

    def updateWeights(self, learning_rate):
        """
        Update the weights of the layer
        """
        self.weights -= learning_rate * (self.__cummulated_gradient / self.__counter)
        self.__resetCummulatedGradient()
