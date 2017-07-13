# -*- coding: utf-8 -*-

import sys
import logging
import copy
import matplotlib.pyplot as plt

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from util.loss_functions import BinaryCrossEntropyError
from model.logistic_layer import LogisticLayer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.1, epochs=200):

        self.learningRate = learningRate
        self.epochs = epochs

        # copy the object to avoid referenzes
        self.trainingSet = copy.copy(train)
        self.validationSet = copy.copy(valid)
        self.testSet = copy.copy(test)

        # Appending the bias
        self.trainingSet.input = np.insert(self.trainingSet.input, self.trainingSet.input.shape[1], 1, axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, self.validationSet.input.shape[1], 1, axis=1)
        self.testSet.input = np.insert(self.testSet.input, self.testSet.input.shape[1], 1, axis=1)

        # Initialize the weight vector with small values
        self.weight = 0.01 * np.random.randn(self.trainingSet.input.shape[1])

        self.bce = BinaryCrossEntropyError()

        self.accuracy_vec = []

        n_hidden_units1 = 1000
        n_hidden_units2 = 28*28

        hidden_layer1 = LogisticLayer(nIn=self.trainingSet.input.shape[1], nOut=n_hidden_units1, activation="sigmoid")
        hidden_layer2 = LogisticLayer(nIn=n_hidden_units1, nOut=n_hidden_units2, activation="sigmoid")
        classifier_layer = LogisticLayer(nIn=n_hidden_units2, nOut=1, isClassifierLayer=True, activation="sigmoid")

        self.layers = [hidden_layer1, hidden_layer2, classifier_layer]

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        self.verbose = verbose

        for epoch in range(0, self.epochs):

            for i in range(0, self.trainingSet.input.shape[0]):
                input = self.trainingSet.input[i, :]
                label = self.trainingSet.label[i]

                y_pred = self.fire(input)

                #output gradient
                dE_dy = (label - y_pred) / ((y_pred - 1).conjugate().dot(y_pred))

                for layer in reversed(self.layers):
                    dE_dy = layer.computeDerivative(dE_dy)

            for layer in self.layers:
                layer.updateWeights()


            if verbose:
                # cross validation accuracy
                y_cv_pred = np.asarray(self.evaluate(self.validationSet.input))
                accuracy = 1.0 - np.mean(np.abs(self.validationSet.label - y_cv_pred))
                self.accuracy_vec += [accuracy]
                print("Epoch [{}/{}]: Cross validation accuracy: {}".format(epoch + 1, self.epochs, accuracy))

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance) > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """

        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return map(self.classify, test)

    #def updateWeights(self, grad):
    #   self.weight += -self.learningRate * grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid

        output = input
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def drawPlot(self):
        if self.verbose:
            plt.plot(range(0, len(self.accuracy_vec)), self.accuracy_vec)
            plt.show()
