# This neural network has been adopted by me from DDO course in TU Ilmenau.
import shutil
import os


import concurrent.futures
import autograd.numpy as np
import numpy as np
import random

import multiprocessing

from NN.optimizer import OptimizerEnum
from NN.optimizer import OptimizerFactory
from NN.optimizer import OptimizerConfiguration


class NeuralNetworkConfiguration:
    def __init__(self,
                 sizes: list,
                 lambda_l1: float,
                 lambda_l2: float,
                 optimizer_configuration: OptimizerConfiguration,
                 mini_batch_size: int,
                 output=False,
                 biases=None,
                 weights=None
                 ):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.optimizer_configuration = optimizer_configuration
        self.max_epochs = 100
        self.mini_batch_size = mini_batch_size
        self.output = output
        self.biases = biases
        self.weights = weights

    def is_trained(self) -> bool:
        if self.biases is None:
            return False

        if self.weights is None:
            return False

        if len(self.biases) != len(self.sizes) - 1:
            return False

        if len(self.weights) != len(self.sizes) - 1:
            return False

        return True


class EpochLog:
    def __init__(self, index: int, W=None, B=None, accuracy=None):
        self.index = index
        self.W: list = W
        self.B: list = B
        self.accuracy = accuracy

    def save(self, root_path, file_format):
        if os.path.exists(root_path):
            shutil.rmtree(root_path)
            os.mkdir(root_path)
        else:
            os.mkdir(root_path)

        file_path = os.path.join(root_path, f'accuracy.{file_format}')
        if file_format == 'txt':
            np.savetxt(file_path, np.array([self.accuracy]), fmt='%.8f')
        else:
            np.save(file_path, self.accuracy)

        layers_number = len(self.W)

        for i in range(layers_number):
            layer_path = os.path.join(root_path, f'layer_{i}')
            if os.path.exists(layer_path):
                shutil.rmtree(layer_path)
                os.mkdir(layer_path)
            else:
                os.mkdir(layer_path)

            W_path = os.path.join(layer_path, f'W.{file_format}')
            B_path = os.path.join(layer_path, f'B.{file_format}')
            if file_format == 'txt':
                np.savetxt(W_path, np.array(self.W[i]), fmt='%.8f')
                np.savetxt(B_path, np.array(self.B[i]), fmt='%.8f')
            else:
                np.save(W_path, np.array(self.W[i]))
                np.save(B_path, np.array(self.B[i]))

    def load(self, root_path, file_format):
        if os.path.exists(root_path):
            self.W = []
            self.B = []

            file_path = os.path.join(root_path, f'accuracy.{file_format}')
            if file_format == 'txt':
                self.accuracy = float(np.loadtxt(file_path))
            else:
                with np.load(file_path) as loaded:
                    self.accuracy = float(loaded['arr_0'])

            files = os.listdir(root_path)
            files.sort()

            for f in files:
                layer_path = os.path.join(root_path, f)

                if os.path.isdir(layer_path):
                    W_path = os.path.join(layer_path, f'W.{file_format}')
                    B_path = os.path.join(layer_path, f'B.{file_format}')

                    if file_format == 'txt':
                        w = np.loadtxt(W_path)
                        self.W.append(w)
                        b = np.matrix(np.loadtxt(B_path)).T
                        self.B.append(b)
                    else:
                        with np.load(W_path) as loaded:
                            self.W.append(loaded['arr_0'])

                        with np.load(B_path) as loaded:
                            self.B.append(loaded['arr_0'])


class TrainingLog:
    def __init__(self, epochs=None, max_accuracy=0.0):
        if epochs is None:
            epochs = []

        self.epochs: [EpochLog] = epochs
        self.max_accuracy = max_accuracy

    def get_best_epoch(self) -> EpochLog:
        for i in range(len(self.epochs)-1, -1, -1):
            if self.epochs[i].accuracy == self.max_accuracy:
                return self.epochs[i]

    def get_last_epoch(self) -> EpochLog:
        return self.epochs[-1]

    def is_trained(self) -> bool:
        return len(self.epochs) > 0 and self.max_accuracy > 0

    def save(self, root_path, file_format):
        if os.path.exists(root_path):
            shutil.rmtree(root_path)
            os.mkdir(root_path)
        else:
            os.mkdir(root_path)

        # file_path = os.path.join(root_path, f'max_accuracy.{file_format}')
        # if file_format == 'txt':
        #     np.savetxt(file_path, np.array([self.max_accuracy]), fmt='%.8f')
        # else:
        #     np.save(file_path, self.max_accuracy)
        #
        # for i in range(len(self.epochs)):
        #     layer_path = os.path.join(root_path, f'epoch_{i}')
        #     self.epochs[i].save(layer_path, file_format)

        best_epoch = self.get_best_epoch()
        best_epoch.save(os.path.join(root_path, 'best_epoch'), 'txt')

    def load(self, root_path, file_format):
        if os.path.exists(root_path):
            self.epochs = []

            # file_path = os.path.join(root_path, f'max_accuracy.{file_format}')
            # if file_format == 'txt':
            #     self.max_accuracy = float(np.loadtxt(file_path))
            # else:
            #     with np.load(file_path) as loaded:
            #         self.max_accuracy = float(loaded['arr_0'])
            #
            # files = os.listdir(root_path)
            # files.sort()
            #
            # i = 0
            # for f in files:
            #     epoch_path = os.path.join(root_path, f)
            #     if os.path.isdir(epoch_path):
            #         epoch = EpochLog(i)
            #         epoch.load(epoch_path, file_format)
            #         self.epochs.append(epoch)
            #         i += 1

            best_epoch_path = os.path.join(root_path, 'best_epoch')
            if os.path.isdir(best_epoch_path):
                best_epoch = EpochLog(0)
                best_epoch.load(best_epoch_path, file_format)
                self.epochs.append(best_epoch)
                self.max_accuracy = best_epoch.accuracy


class NeuralNetwork:
    """
    Class representing a feedforward neural network
    """

    def __init__(self, configuration: NeuralNetworkConfiguration):
        """
        Initializes an instance of the class
        Arguments:
            sizes: a list containing numbers of neurons in each layer
            lambda_l1: L1 regularization parameter
            lambda_l2: L2 regularization parameter
            output: setting to True enables summary after each epoch
        """
        self._configuration = configuration
        self.training_log = TrainingLog()

        if not self._configuration.is_trained():
            # Randomly initialize weights and biases using standard distribution
            self._configuration.biases = [np.random.randn(y, 1) for y in self._configuration.sizes[1:]]
            self._configuration.weights = [np.random.randn(y, x)
                                           for x, y in zip(self._configuration.sizes[:-1], self._configuration.sizes[1:])]

    def is_trained(self) -> bool:
        return self.training_log.is_trained()

    def _can_load(self, root_path) -> bool:
        return os.path.exists(root_path)

    def _save(self, root_path):
        self.training_log.save(root_path, 'txt')

    def _load(self, root_path):
        self.training_log.load(root_path, 'txt')
        epoch = self.training_log.get_best_epoch()
        self._configuration.weights = epoch.W
        self._configuration.biases = epoch.B

    def sigmoid(self, a):
        """
        Computes a sigmoid function
        Arguments:
            a: an argument of the function
        Returns:
            A sigmoid value of the function
        """

        a = np.clip(a, -500, 500)
        return 1 / (1 + np.exp(-a))

    def sigmoid_derivative(self, sigmoid_a):
        """
        Computes a derivative of a sigmoid function
        Arguments:
            sigmoid_a: sigmoid value of the function a
        Returns:
            A sigmoid derivative value of the function a
        """

        return np.multiply(sigmoid_a, (1 - sigmoid_a))

    def forward_pass(self, x, return_lists=False):
        """
        Performs forward propagation
        Arguments:
            x: a sample (column-vector)
            return_lists: set False to return the last layer output or set to
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
        for l in range(1, self._configuration.num_layers):
            # ToDo: Compute the weighted input of the layer (0.5 points)
            z_l = self._configuration.weights[l - 1] @ a[l - 1] + self._configuration.biases[l - 1]

            # ToDo: Compute the activation function (output) of the layer
            #       (0.5 points)
            a_l = self.sigmoid(z_l)

            # ToDo: Compute the activation function derivative (0.5 points)
            a_I_l = self.sigmoid_derivative(a_l)

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
        for l in range(self._configuration.num_layers - 2, 0, -1):
            # ToDo: Compute the error gradient of the layer (0.5 points)
            # $$\delta^l=a^{l(I)}\odot(W^{l+1})^T \delta^{l+1}.$$
            delta_l = np.multiply(a_I[l], self._configuration.weights[l].T @ delta_l)

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

    def update_mini_batch(self, mini_batch, optimizer_configuration: OptimizerConfiguration):
        """
        Updates the network parameters based on a single mini-batch
        Arguments:
            mini_batch: a mini-batch
            optimizer: an optimizer
        """

        # Initialize the list of partial derivatives of the cost function w.r.t.
        # the network weights of each layer
        dJdW = [np.zeros(W.shape) for W in self._configuration.weights]

        # Initialize the list of partial derivatives of the cost function w.r.t.
        # the network biases of each layer
        dJdb = [np.zeros(b.shape) for b in self._configuration.biases]

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
        L1 = [self._configuration.lambda_l1 * np.sign(W_l) for W_l in self._configuration.weights]
        L2 = [self._configuration.lambda_l2 * W_l for W_l in self._configuration.weights]
        dJdW = [dJdW_l + L1_l + L2_l for dJdW_l, L1_l, L2_l in zip(dJdW, L1, L2)]

        optimizer = OptimizerFactory.instance(optimizer_configuration)

        biases = optimizer.update(self._configuration.biases, dJdb, 'b')
        weights = optimizer.update(self._configuration.weights, dJdW, 'W')

        return biases, weights

    def train(self, training_data,
              test_data=None, random_shuffle=True):
        """
        Trains the neural network
        Arguments:
            training_data: a training set
            test_data: a test set
            random_shuffle: determines if the dataset needs to be shuffled
        """
        # The test set size (if provided)
        if test_data is not None:
            n_test = len(test_data)

        # The training set size
        n = len(training_data)

        # Initialize the auxiliary variables
        max_accuracy = 0.1
        max_no_improvement_epochs = 5
        no_improvement_epochs_counter = 0

        # Iterate over epochs
        for j in range(self._configuration.max_epochs):
            # Shuffle the training set (step 2)
            if random_shuffle:
                random.shuffle(training_data)

            # Divide the training set into mini batches, creating list of mini
            # batches with non-overlapping data (step 2)
            mini_batches = [training_data[k:k + self._configuration.mini_batch_size]
                            for k in range(0, n, self._configuration.mini_batch_size)]


            # new_biases = []
            # new_weights = []
            #
            # for i in range(self.__configuration.num_layers - 1):
            #     b_i = np.zeros(self.__configuration.biases[i].shape)
            #     new_biases.append(b_i)
            #     w_i = np.zeros(self.__configuration.weights[i].shape)
            #     new_weights.append(w_i)
            #
            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     # Use list comprehension to execute the function concurrently
            #     results = [executor.submit(self.update_mini_batch, mini_batch, self.__configuration.optimizer_configuration) for mini_batch in mini_batches]
            #
            #     for future in concurrent.futures.as_completed(results):
            #         b, w = future.result()
            #         for i in range(self.__configuration.num_layers - 1):
            #             new_biases[i] = new_biases[i] + b[i]
            #             new_weights[i] = new_weights[i] + w[i]
            #
            # for i in range(self.__configuration.num_layers - 1):
            #     new_biases[i] = new_biases[i] / len(mini_batches)
            #     new_weights[i] = new_weights[i] / len(mini_batches)
            #
            # self.__configuration.biases = new_biases
            # self.__configuration.weights = new_weights


            # Iterate over the mini-batches (step 3)
            for mini_batch in mini_batches:
                # Update the network parameters based on a mini-batch
                b, w = self.update_mini_batch(mini_batch, self._configuration.optimizer_configuration)
                self._configuration.biases = b
                self._configuration.weights = w

            success_training = self.evaluate(training_data)
            train_accuracy = success_training / len(training_data)

            success_tests = self.evaluate(test_data)
            test_accuracy = success_tests / n_test

            # Print the epoch summary
            if test_data is not None and self._configuration.output:
                print(f'Epoch {j}: train {success_training} / {len(training_data)} = {train_accuracy}; tests {success_tests} / {n_test} = {test_accuracy}')
            elif self._configuration.output:
                print('Epoch {0} is complete'.format(j + 1))

            # Stop training if 100% accuracy is achieved
            if test_data is not None:
                epoch_log = EpochLog(j, self._configuration.weights, self._configuration.biases, test_accuracy)
                self.training_log.epochs.append(epoch_log)

                if test_accuracy > max_accuracy:
                    no_improvement_epochs_counter = 0
                    max_accuracy = test_accuracy
                    self.training_log.max_accuracy = max_accuracy
                else:
                    no_improvement_epochs_counter += 1
                    if no_improvement_epochs_counter >= max_no_improvement_epochs:
                        print('max_no_improvement_epochs reached')
                        break

                if test_accuracy == 1.0:
                    print('100% accuracy')
                    break

        # Print training summary
        if test_data is not None:
            print('Current accuracy:', test_accuracy)
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
