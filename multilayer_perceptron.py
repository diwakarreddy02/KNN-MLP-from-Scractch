# # multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
# #
# # Submitted by: [enter your full name here] -- [enter your IU username here]
# #
# # Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self.classlist=np.array(list(set(y)))
        self._y = one_hot_encoding(y)
        self.n_features=self._X.shape[1]
        self.n_out_Values=self._y.shape[1]
        np.random.seed(42)
        

    def initialize_attributes(self) :
        self._h_weights = 0.1*np.random.rand(self.n_features,self.n_hidden)
        self._h_bias = 0.1*np.random.rand(1,self.n_hidden)
        self._o_weights = 0.1*np.random.rand(self.n_hidden,self.n_out_Values)
        self._o_bias = 0.1*np.random.rand(1,self.n_out_Values)
        
        
    def feed_forward(self) :
        self.forward_temp=np.dot(self._X,self._h_weights)+self._h_bias
        self.forward_temp1=self.hidden_activation(self.forward_temp)
        self.forward_temp2=np.dot(self.forward_temp1,self._o_weights)+self._o_bias
        self.ypred=self._output_activation(self.forward_temp2)

    def error_derivate(self) :
        curr_error=self.ypred-self._y
        delay_output=curr_error*self._output_activation(self.forward_temp2,derivative=True)
        derivative_output=np.dot(self.forward_temp1.T,delay_output)
        return (derivative_output,delay_output)


    def weight_update_output(self,derivative_output,delay_output) :
        self._o_weights=self._o_weights-self.learning_rate*derivative_output
        self._o_bias=self._o_bias-self.learning_rate*np.sum(delay_output,axis=0)

    def error_derivate_hidden(self,derivative_output,delay_output) :
        delay_hidden=np.dot(delay_output,self._o_weights.T)*self.hidden_activation(self.forward_temp,derivative=True)
        derivative_hidden=np.dot(self._X.T,delay_hidden)
        return (derivative_hidden,delay_hidden)

    def weight_update_hidden(self,derivative_output,delay_output,derivative_hidden,delay_hidden) :
         self._h_weights=self._h_weights-self.learning_rate*derivative_hidden
         self._h_bias=self._h_bias-self.learning_rate*np.sum(derivative_hidden,axis=0)



    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)
        self.initialize_attributes()
        for iteration in range(self.n_iterations):
            #Data is fed into forward iteration
            self.feed_forward()

            
            #Checks the errors and derivatives
            derivative_output,delay_output=self.error_derivate()
            #Updates the weights for output layer
            self.weight_update_output(derivative_output,delay_output)

            #Checks erros and derivatives for hidden layer
            derivative_hidden,delay_hidden=self.error_derivate_hidden(derivative_output,delay_output)
            #Updates weights for hidden layer
            self.weight_update_hidden(derivative_output,delay_output,derivative_hidden,delay_hidden)

        return
            

           
        
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        #Multiplying given training dataset 'X' with weights assumed
        product=np.dot(X,self._h_weights)
        #Adding bias to the product 
        product_bias=product+self._h_bias 
        #Passing through activation function for hidden layer 
        activated_hidden=self.hidden_activation(product_bias)
        #multiplying with o_weights
        hidden_weighted=np.dot(activated_hidden,self._o_weights)
        #Adding o_bias 
        output=hidden_weighted+self._o_bias

        answer = []
        for i in np.argmax(output,axis=1) :
            answer.append(self.classlist[i])

        return np.array(answer)
