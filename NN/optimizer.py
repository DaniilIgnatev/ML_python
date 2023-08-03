import numpy as np
import numpy.linalg as la
from enum import Enum


class OptimizerEnum(Enum):
    GDOptimizer = 'GDOptimizer'



class OptimizerConfiguration:
    """
    Creates an instance of the optimizer configuration
    Arguments:
        alpha: a step size
        max_iter: a maximum number of iterations
        eps: a tolerance value
        record: a flag for recording optimization history
    """
    def __init__(self, name: OptimizerEnum, alpha, max_iter, eps, record):
        self.name = name
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
        self.record = record


class Optimizer:
    """
    Optimizer
    """

    def __init__(self, configuration: OptimizerConfiguration):
        """
        Creates an instance of the optimizer
        Arguments:
            configuration - container with parameters
        Returns:
            Optimizer
        """
        self.configuration = configuration

        self.iter = None
        self.last_deltas = {}
        self.history = {}

    def _deltas(self, grads, key):
        """
        Computes the deltas based on the gradients
        Arguments:
            values: a list containing the current values of the parameter
            grads: a list containing the current gradients of the parameter
            key: a key identifying the type of the parameter
        Returns:
            The list containing the updated values of the parameter
        """
        pass

    def update(self, values, grads, key='default'):
        """
        Updates the parameters
        Arguments:
            values: a list containing the current values of the parameter
            grads: a list containing the current gradients of the parameter
            key: a key identifying the type of the parameter
        Returns:
            The list containing the updated values of the parameter
        """
        # Wrap the values and deltas if necessary
        if type(values) is not list:
            values = [values]
        if type(grads) is not list:
            grads = [grads]

        # Compute the deltas
        deltas = self._deltas(grads, key)

        # Compute the updated values
        # ToDo:
        # Hint: use list comprehension and zip
        updated_values = [value + delta for value, delta in zip(values, deltas)]

        # Store the updated values if necessary
        if self.configuration.record:
            if key not in self.history:
                self.history[key] = [values]
            self.history[key].append(updated_values)

        # Unwrap the updated values if necessary
        if len(updated_values) < 2:
            updated_values = updated_values[0]

        # Return the updated values
        return updated_values

    def run(self, key='default'):
        """
        Updates the iteration counter and checks the termination criteria
        Returns:
            False if the optimization should be terminated, and True otherwise
        """
        # Initialize the iteration counter
        if self.iter is None:
            self.iter = 0
            return True

        # Increment the iteration counter
        self.iter = self.iter + 1

        # Check the iteration counter
        if self.iter > self.configuration.max_iter - 1:
            return False

        # Compute the delta norm
        delta_norm = la.norm(self.last_deltas[key][0])

        # Check the delta norm
        if delta_norm < self.configuration.eps:
            return False

        return True

    def get_iter(self):
        """
        Returns the iteration counter
        Returns:
            The iteration counter
        """
        return self.iter

    def get_history(self, key='default'):
        """
        Returns the optimization history
        Arguments:
            key: a key identifying the type of the parameter
        Returns:
            The optimization history for the parameter with a specified key
        """
        # Retrieve the history
        history = [np.array([h[i] for h in self.history[key]]) for i in range(0, len(self.history[key][0]))]

        # Unwrap the history if necessary
        if len(history) < 2:
            history = history[0]

        # Return the history
        return history


class GDOptimizer(Optimizer):
    """
    Gradient descent optimizer
    """

    def __init__(self, configuration: OptimizerConfiguration):
        """
        Creates an instance of the gradient descent optimizer
        Arguments:
            configuration - container with parameters
        Returns:
            Gradient descent optimizer
        """
        self.configuration = configuration
        super().__init__(self.configuration)

    def _deltas(self, grads, key):
        """
        Computes the deltas based on the gradients
        Arguments:
            grads: a list containing the current gradients of the parameter
            key: a key identifying the type of the parameter
        Returns:
            The list containing the deltas of the parameter
        """
        # TODO: Compute the deltas for gradient descent
        # Hint: use list comprehension
        deltas = [-self.configuration.alpha * grad for grad in grads]

        # Store the deltas
        self.last_deltas[key] = deltas

        # Return the deltas
        return deltas


class OptimizerFactory:
    @staticmethod
    def instance(configuration: OptimizerConfiguration) -> GDOptimizer | None:
        if configuration.name == OptimizerEnum.GDOptimizer:
            return GDOptimizer(configuration)

        return None
