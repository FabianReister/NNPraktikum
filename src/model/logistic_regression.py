# -*- coding: utf-8 -*-

import sys
import logging
import copy
import matplotlib.pyplot as plt

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


        #copy the object to avoid referenzes
        self.trainingSet = copy.copy(train)
        self.validationSet = copy.copy(valid)
        self.testSet = copy.copy(test)


        #Appending the bias
        self.trainingSet.input = np.insert(self.trainingSet.input,  self.trainingSet.input.shape[1], 1, axis = 1)
        self.validationSet.input = np.insert(self.validationSet.input,  self.validationSet.input.shape[1], 1, axis = 1)
        self.testSet.input = np.insert(self.testSet.input,  self.testSet.input.shape[1], 1, axis = 1)

        # Initialize the weight vector with small values
        self.weight = 0.01 * np.random.randn(self.trainingSet.input.shape[1])

        self.bce = BinaryCrossEntropyError()

        self.accuracy_vec = []

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import DifferentError
        loss = DifferentError()

        learned = False
        iteration = 0

        while not learned:
            grad = 0
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                output = self.fire(input)
                # compute gradient
                grad += -(label - output)*input

                # compute recognizing error, not BCE
                predictedLabel = self.classify(input)
                error = loss.calculateError(label, predictedLabel)
                totalError += error

            self.updateWeights(grad)
            totalError = abs(totalError)
            
            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)
                

            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

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
        listEvaluation = np.asarray(map(self.classify, test))
        for i in range(0,listEvaluation.size):
            if(listEvaluation[i] > 0.5):
                listEvaluation[i] = 1
            else:
                listEvaluation[i] = 0

        return list(listEvaluation)
                    

    def updateWeights(self, grad):
        self.weight -= self.learningRate*grad

        self.weight += -self.learningRate * grad

    def fire(self, input):
        return Activation.sigmoid(np.dot(np.array(input), self.weight))

    def drawPlot(self):
        plt.plot(range(0,len(self.accuracy_vec)) , self.accuracy_vec )
        plt.show()
