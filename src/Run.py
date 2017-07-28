#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron as MLP

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot

import numpy as np

def normalize_magic(x):
    return x - 0.5

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)

    print data.trainingSet.label[0]

    data.trainingSet.input = np.asarray(map(normalize_magic, data.trainingSet.input))
    data.validationSet.input = np.asarray(map(normalize_magic, data.validationSet.input))
    data.testSet.input = np.asarray(map(normalize_magic, data.testSet.input))

    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    
    #myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                    data.validationSet,
    #                                    data.testSet,
    #                                    learningRate=0.005,
    #                                    epochs=30)
                                        
    #myLRClassifier = LogisticRegression(data.trainingSet,
    #                                    data.validationSet,
    #                                    data.testSet,
    #                                    learningRate=0.005,
    #                                    epochs=30)


    myMLPClassifier = MLP(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=10,
                                        epochs=30)
    
    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")

    print("\nPerceptron has been training..")
    #myPerceptronClassifier.train()
    print("Done..")
    
    print("\nLogistic Regression has been training..")
    #myLRClassifier.train()
    print("Done..")

    print("\nMLP Regression ..")
    myMLPClassifier.train()
    print("Done..")

    #exit()

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    #perceptronPred = myPerceptronClassifier.evaluate()
    mlpPred = myMLPClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    #evaluator.printComparison(data.testSet, stupidPred)
    #evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    #evaluator.printComparison(data.testSet, perceptronPred)
    #evaluator.printAccuracy(data.testSet, perceptronPred)
    
    print("\nResult of the MLP recognizer:")
    #evaluator.printComparison(data.testSet, mlpPred)

    y_true = map(np.argmax, data.testSet.label)

    print len(y_true)

    data.testSet.label = y_true
    y_pred = np.asarray(mlpPred).ravel()
    print y_pred.shape

    evaluator.printAccuracy(data.testSet, y_pred)
    
    # Draw
    #plot = PerformancePlot("Logistic Regression validation")
    #plot.draw_performance_epoch(myMLPClassifier.performances,
    #                            myMLPClassifier.epochs)
    
    
if __name__ == '__main__':
    main()
