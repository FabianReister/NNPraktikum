# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from util.loss_functions import BinaryCrossEntropyError

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

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test


        # Initialize the weight vector with small values
        self.weight = 0.01 * np.random.randn(self.trainingSet.input.shape[1])

        self.bce = BinaryCrossEntropyError()

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        

        for epoch in range(0, self.epochs):

            y_pred = np.asarray(map(self.classify, self.trainingSet.input))

            bce_error = self.bce.calculateError(np.asarray(self.trainingSet.label), y_pred)


            dE_dy = (self.trainingSet.label - y_pred ) / ( ( y_pred - 1 ).conjugate().dot( y_pred ) )  
            dy_dx = Activation.sigmoidPrime(y_pred)
            dx_dw = y_pred

            dE_dw = np.asarray(dE_dy * dy_dx * dx_dw)
            
            grad = self.trainingSet.input.transpose().dot( dE_dw )

            self.updateWeights( grad ) 
            
            
            if verbose:

                # cross validation accuracy
                y_cv_pred = np.asarray(self.evaluate(self.validationSet.input))
                accuracy = 1.0 - np.mean(np.abs(self.validationSet.label - y_cv_pred))
                print("Epoch [{}/{}]: Cross validation accuracy: {}".format(epoch+1, self.epochs, accuracy))


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
        return self.fire(testInstance)

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
        listEvaluation = np.asarray(map(self.classify, test))
        for i in range(0,listEvaluation.size):
            if(listEvaluation[i] > 0.5):
                listEvaluation[i] = 1
            else:
                listEvaluation[i] = 0

        return list(listEvaluation)
                    

    def updateWeights(self, grad):

        self.weight += -self.learningRate * grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))

