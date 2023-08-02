import numpy as np
import numpy.linalg as la


class Optimizer:
    """
    Optimizer
    """

    def __init__(self, alpha, max_iter, eps, record):
        """
        Creates an instance of the optimizer
        Arguments:
            alpha: a step size
            max_iter: a maximum number of iterations
            eps: a tolerance value
            record: a flag for recording optimization history
        Returns:
            Optimizer
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
        self.record = record
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
        if self.record:
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
        if self.iter > self.max_iter - 1:
            return False

        # Compute the delta norm
        delta_norm = la.norm(self.last_deltas[key][0])

        # Check the delta norm
        if delta_norm < self.eps:
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


class GDoptimizer(Optimizer):
    """
    Gradient descent optimizer
    """

    def __init__(self, alpha, max_iter, eps, record=False):
        """
        Creates an instance of the gradient descent optimizer
        Arguments:
            alpha: a step size
            max_iter: a maximum number of iterations
            eps: a tolerance value
            record: a flag for recording optimization history
        Returns:
            Gradient descent optimizer
        """
        super().__init__(alpha, max_iter, eps, record)

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
        deltas = [-self.alpha * grad for grad in grads]

        # Store the deltas
        self.last_deltas[key] = deltas

        # Return the deltas
        return deltas
