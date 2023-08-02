import autograd.numpy as np
import numpy as np
import random

from NN.optimizer import Optimizer

class NeuralNetwork:
    """
    Class representing a feedforward neural network
    """

    def __init__(self, sizes, lambda_l1=0, lambda_l2=0, output=True):
        """
        Initializes an instance of the class
        Arguments:
            sizes: a list containing numbers of neurons in each layer
            lambda_l1: L1 regularization parameter
            lambda_l2: L2 regularization parameter
            output: setting to True enables summary after each epoch
        """
        # Store the number of layers and their sizes
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Randomly initialize weights and biases using standard distribution
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # Store regularization parameters and the output flag
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.output = output

    def forward_pass(self, x, return_lists=False):
        """
        Performs forward propagation
        Arguments:
            x: a sample (column-vector)
            return_lists: set to False to return the last layer output or set to
                          True to return the lists of the activation function
                          and its derivative values
        Returns:
            The return value depends on the return_lists argument
        """
        # Initialize the list of the activation function values (or output values)
        # for each layer
        a = []

        # Initialize the list of the activation function derivative values for
        # each layer
        a_I = []

        # The first layer does not have an activation function, but it has output
        # being equal to x
        a.append(x)

        # The first layer does not have an activation function derivative, because
        # it does not have an activation function
        a_I.append(None)

        # Iterate over the layers starting from the second one
        for l in range(1, self.num_layers):
            # ToDo: Compute the weighted input of the layer (0.5 points)
            z_l = self.weights[l - 1] @ a[l - 1] + self.biases[l - 1]

            # ToDo: Compute the activation function (output) of the layer
            #       (0.5 points)
            a_l = 1 / (1 + np.exp(-z_l))

            # ToDo: Compute the activation function derivative (0.5 points)
            a_I_l = a_l * (1 - a_l)

            # Append the activation function and its derivative values to the lists
            a.append(a_l)
            a_I.append(a_I_l)

        # Return the lists if necessary
        if return_lists:
            return a, a_I

        # Otherwise, return the last layer output
        return a[-1]

    def backward_pass(self, a, a_I, y):
        """
        Performs backpropagation
        Arguments:
            a: a list of activation function values of each layer
            a_I: a list of activation function derivative values of each layer
            y: a ground truth (scalar)
        Returns:
            The lists of partial derivatives of the loss function w.r.t. the
            network weights and biases of each layer
        """
        # Initialize the list of partial derivatives of the loss function w.r.t.
        # the network weights of each layer
        dJdW = []

        # Initialize the list of partial derivatives of the loss function w.r.t.
        # the network biases of each layer
        dJdb = []

        # ToDo: Compute the last layer (output layer) error gradient (0.5 points)
        delta_l = a[-1] - y

        # ToDo: Compute the partial derivative of the loss function w.r.t. the
        #       last layer weigths (0.5 points)
        dJdW_l = delta_l @ a[-2].T

        # ToDo: Compute the partial derivative of the loss function w.r.t. the
        # last layer bias (0.5 points)
        dJdb_l = delta_l

        # Prepend the partial derivatives to the lists
        dJdW.insert(0, dJdW_l)
        dJdb.insert(0, dJdb_l)

        # Iterate backwards over the layers starting from the second-to-last one
        for l in range(self.num_layers - 2, 0, -1):
            # ToDo: Compute the error gradient of the layer (0.5 points)
            # $$\delta^l=a^{l(I)}\odot(W^{l+1})^T \delta^{l+1}.$$
            delta_l = np.multiply(a_I[l], self.weights[l].T @ delta_l)

            # $$\frac{\partial J^{(i)}}{\partial W^l}=\delta^{l(i)}(a^{l-1{(i)}})^T,\tag{0.4}$$
            # ToDo: Compute the partial derivative of the loss function w.r.t. the
            #       layer weigths (0.5 points)
            dJdW_l = delta_l @ a[l - 1].T

            # $$\frac{\partial J^{(i)}}{\partial b^l}=\delta^{l(i)},\tag{0.5}$$
            # ToDo: Compute the partial derivative of the loss function w.r.t. the
            #       layer bias (0.5 points)
            dJdb_l = delta_l

            # Prepend the partial derivatives to the lists
            dJdW.insert(0, dJdW_l)
            dJdb.insert(0, dJdb_l)

        # Return the lists
        return dJdW, dJdb

    def update_mini_batch(self, mini_batch, optimizer: Optimizer):
        """
        Updates the network parameters based on a single mini-batch
        Arguments:
            mini_batch: a mini-batch
            optimizer: an optimizer
        """
        # Initialize the list of partial derivatives of the cost function w.r.t.
        # the network weights of each layer
        dJdW = [np.zeros(W.shape) for W in self.weights]

        # Initialize the list of partial derivatives of the cost function w.r.t.
        # the network biases of each layer
        dJdb = [np.zeros(b.shape) for b in self.biases]

        # Compute partial derivatives of the cost function by summing up
        # partial derivatives of the loss function evaluated on the mini-batch
        for x, y in mini_batch:
            # Perform forward propagation on the sample
            a, a_I = self.forward_pass(x, return_lists=True)

            # Perform backpropagation on the sample
            dJdW_i, dJdb_i = self.backward_pass(a, a_I, y)

            # Append the partial derivatives to the sums of partial derivatives
            dJdW = [dJdW_l + dJdW_l_i for dJdW_l, dJdW_l_i in zip(dJdW, dJdW_i)]
            dJdb = [dJdb_l + dJdb_l_i for dJdb_l, dJdb_l_i in zip(dJdb, dJdb_i)]

        # Divide the sums by the mini-batch size
        dJdW = [1 / len(mini_batch) * dJdW_l for dJdW_l in dJdW]
        dJdb = [1 / len(mini_batch) * dJdb_l for dJdb_l in dJdb]

        # ToDo: Apply the L1 and L2 regularizations by modifying the cost
        #       function partial derivatives (0.5 points)
        L1 = [self.lambda_l1 * np.sign(dJdW_l) for dJdW_l in dJdW]
        L2 = [self.lambda_l2 * dJdW_l for dJdW_l in dJdW]
        dJdW = [dJdW_l + L1_l + L2_l for dJdW_l, L1_l, L2_l in zip(dJdW, L1, L2)]

        # Update the network parameters using an external optimizer
        self.weights = optimizer.update(self.weights, dJdW, 'W')
        self.biases = optimizer.update(self.biases, dJdb, 'b')

    def train(self, training_data, epochs, mini_batch_size, optimizer: Optimizer,
              test_data=None, random_shuffle=True):
        """
        Trains the neural network
        Arguments:
            training_data: a training set
            epochs: a maximum number of epochs
            mini_batch_size: a mini-batch size
            optimizer: an optimizer
            test_data: a test set
            random_shuffle: determines if the dataset needs to be shuffled
        """
        # The test set size (if provided)
        if test_data is not None:
            n_test = len(test_data)

        # The training set size
        n = len(training_data)

        # Initialize the auxiliary variables
        success_tests = 0
        max_accuracy = 0.1

        # Iterate over epochs
        for j in range(epochs):
            # Shuffle the training set (step 2)
            if random_shuffle:
                random.shuffle(training_data)

            # Divide the training set into mini batches, creating list of mini
            # batches with non-overlapping data (step 2)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            # Iterate over the mini-batches (step 3)
            for mini_batch in mini_batches:
                # Update the network parameters based on a mini-batch
                self.update_mini_batch(mini_batch, optimizer)

            # Print the epoch summary
            if test_data is not None and self.output:
                success_tests = self.evaluate(test_data)
                print('Epoch {0}: {1} / {2}'.format(
                    j + 1, success_tests, n_test))
            elif self.output:
                print('Epoch {0} is complete'.format(j + 1))

            # Stop training if 100% accuracy is achieved
            if test_data is not None:
                accuracy = success_tests / n_test
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                if accuracy == 1.0:
                    print('100% accuracy')
                    break

        # Print training summary
        if test_data is not None:
            print('Current accuracy:', accuracy)
            print('Maximum accuracy:', max_accuracy)

    def evaluate(self, test_data):
        """
        Evaluates the neural network on a given dataset
        Arguments:
            test_data: a test set
        Returns:
            Number of successfully predicted samples from the test set
        """
        # Compute neural network outputs
        test_results = [(self.forward_pass(x) >= 0.5, y) for (x, y) in test_data]

        # Return the number of successfully predicted samples
        return sum(int(x == y) for (x, y) in test_results)
