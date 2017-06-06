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

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        """
        TODO
        implement as function argument 
        """
        self.batch_size = 10

        # Initialize the weight vector with small values
        self.weight = 0.01 * np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        

        for epoch in range(1, self.epochs):
            for counter in range(0, self.trainingSet.input.shape[0]/10):
                grad = []
                y_pred = np.asarray(map(self.classify, self.trainingSet.input[counter*10:counter*10+9]))
                bce = BinaryCrossEntropyError()
                bce_error = bce.calculateError(np.asarray(self.trainingSet.label[counter*10:counter*10+9]), y_pred)
                grad = bce_error * self.trainingSet.input[counter*10:counter*10+9]
                self.updateWeights( np.sum(grad, axis=0) ) 

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

        print(grad)
        self.weight +=  self.learningRate * grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))

