# # utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
# #
# # Submitted by: Venkata Diwakar Reddy Kashireddy -- vkashir
# #
# # Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff


import numpy as np
from numpy.core.fromnumeric import mean
import math


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    e_dist = np.sqrt(np.sum(np.square(x1-x2)))
    return e_dist


    # raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    m_dist = 0
    for a, b in zip(x1, x2):
        absolute = abs(a-b)
        m_dist += absolute
    return m_dist

    # raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if not derivative:
        return x
    else:
        return np.ones(np.shape(x))
    # raise NotImplementedError('This function must be implemented by the student.')
    


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
   
    answer=1+np.exp(-x)
    answer=1/answer
    if not derivative:
        return answer
    else:
        return answer*(1-answer)


    # raise NotImplementedError('This function must be implemented by the student.')


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if not derivative:
        return np.tanh(x)
    else:

        return (1-(tanh(x)**2))
    # raise NotImplementedError('This function must be implemented by the student.')


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    #     if not derivative:
#         return np.maximum(x,0)
#     else:
#         for i in x:
#             if i>0:
#                 i = 1
#             elif i<=0:
#                 i = 0
#     return x
    d=np.copy(x)
    
    if not derivative:
        res= np.maximum(d,0)
    else:
        d[d<=0]=0
        d[d>0]=1
        res=d
    return res


    # raise NotImplementedError('This function must be implemented by the student.')


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    ce_temp = y*p
    ce_temp = ce_temp[ce_temp!=0]
    ce_loss = np.mean(-np.log(ce_temp))
    return ce_loss

      # raise NotImplementedError('This function must be implemented by the student.')


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """

    oharray = [[0 for j in range(np.max(y)+1)] for i in range(y.shape[0])]
    for i in range(y.shape[0]):
       oharray[i][y[i]] = 1
    return np.array(oharray)


    # raise NotImplementedError('This function must be implemented by the student.')
