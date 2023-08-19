import numpy as np
import numpy.linalg as la
from enum import Enum


class OptimizerEnum(Enum):
    GDOptimizer = 'GDOptimizer'
    AdaGradOptimizer = 'AdaGradOptimizer'
    RMSPropOptimizer = 'RMSPropOptimizer'
    MomentumOptimizer = 'MomentumOptimizer'
    ADAMOptimizer = 'ADAMOptimizer'


class OptimizerConfiguration:
    """
    Creates an instance of the optimizer configuration
    Arguments:
        alpha: a step size
        max_iter: a maximum number of iterations
        eps: a tolerance value
        record: a flag for recording optimization history
    """
    def __init__(self, name: OptimizerEnum, alpha, beta1=None, beta2=None, max_iter=5000, eps=1e-3, record=False):
        self.name = name
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
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
        self._configuration = configuration
        self.Lambda = 1e-8

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
        if self._configuration.record:
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
        if self.iter > self._configuration.max_iter - 1:
            return False

        # Compute the delta norm
        delta_norm = la.norm(self.last_deltas[key][0])

        # Check the delta norm
        if delta_norm < self._configuration.eps:
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
        super().__init__(configuration)

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
        deltas = [-self._configuration.alpha * grad for grad in grads]

        # Store the deltas
        self.last_deltas[key] = deltas

        # Return the deltas
        return deltas


class AdaGradOptimizer(Optimizer):
    """
    AdaGrad optimizer
    """

    def __init__(self, configuration: OptimizerConfiguration):
        """
        Initializes an instance of the gradient descent optimizer with momentum
        Arguments:
            configuration - container with parameters
        """
        super().__init__(configuration)
        self.nu = {}

    def _deltas(self, grads, key):
        """
        Computes the deltas based on the gradients
        Arguments:
            grads: a list containing the current gradients of the parameter
            key: a key identifying the type of the parameter
        Returns:
            The list containing the deltas of the parameter
        """
        # Initialize the sums of squared partial derivatives
        if key not in self.nu:
            self.nu[key] = [np.zeros(grad.shape) for grad in grads]

        # ToDo: Update the sums of squared partial derivatives (1 point)
        self.nu[key] = [nu + (grad * grad) for nu, grad in zip(self.nu[key], grads)]

        # ToDo: Compute the deltas (1 point)
        deltas = [-self._configuration.alpha / np.sqrt(self.Lambda + nu) * g for nu,g in zip(self.nu[key], grads)]

        # Store the updated deltas
        self.last_deltas[key] = deltas

        # Return the computed deltas
        return deltas


class RMSPropOptimizer(Optimizer):
    """
    RMSProp optimizer
    """

    def __init__(self, configuration: OptimizerConfiguration):
        """
        Initializes an instance of the gradient descent optimizer with momentum
        Arguments:
            configuration - container with parameters
        """
        super().__init__(configuration)
        self.u = {}

    def _deltas(self, grads, key):
        """
        Computes the deltas based on the gradients
        Arguments:
            grads: a list containing the current gradients of the parameter
            key: a key identifying the type of the parameter
        Returns:
            The list containing the deltas of the parameter
        """
        # Initialize the last deltas
        if key not in self.u:
            self.u[key] = [np.zeros(grad.shape) for grad in grads]

        # ToDo: Update the exponential averages of squared partial derivatives (1 point)
        self.u[key] = [self._configuration.beta1 * u + ((1 - self._configuration.beta1) * (grad * grad)) for u, grad in zip(self.u[key], grads)]

        # ToDo: Compute the deltas (1 point)
        deltas = [-self._configuration.alpha / np.sqrt(self.Lambda + u) * g for u, g in zip(self.u[key], grads)]

        # Store the updated deltas
        self.last_deltas[key] = deltas

        # Return the computed deltas
        return deltas


class MomentumOptimizer(Optimizer):
    """
    RMSProp optimizer
    """

    def __init__(self, configuration: OptimizerConfiguration):
        """
        Initializes an instance of the gradient descent optimizer with momentum
        Arguments:
            configuration - container with parameters
        """
        super().__init__(configuration)
        self.m = {}

    def _deltas(self, grads, key):
        """
        Computes the deltas based on the gradients
        Arguments:
            grads: a list containing the current gradients of the parameter
            key: a key identifying the type of the parameter
        Returns:
            The list containing the deltas of the parameter
        """
        if key not in self.m:
            self.m[key] = [np.zeros(grad.shape) for grad in grads]

        self.m[key] = [self._configuration.beta1 * m + (1 - self._configuration.beta1) * grad for m, grad in zip(self.m[key], grads)]

        deltas = [-self._configuration.alpha * m for m in self.m[key]]
        self.last_deltas[key] = deltas

        return deltas


class AdamOptimizer(Optimizer):
    def __init__(self, configuration: OptimizerConfiguration):
        """
        Initializes an instance of the gradient descent optimizer with momentum
        Arguments:
            configuration - container with parameters
        """
        super().__init__(configuration)
        self.m = {}
        self.u = {}

    def _deltas(self, grads, key):
        """
        Computes the deltas based on the gradients
        Arguments:
            grads: a list containing the current gradients of the parameter
            key: a key identifying the type of the parameter
        Returns:
            The list containing the deltas of the parameter
        """
        if key not in self.m:
            self.m[key] = [np.zeros(grad.shape) for grad in grads]

        self.m[key] = [self._configuration.beta1 * m + (1 - self._configuration.beta1) * grad for m, grad in zip(self.m[key], grads)]
        m_corrected = [m / (1 - self._configuration.beta1) for m in self.m[key]]

        if key not in self.u:
            self.u[key] = [np.zeros(grad.shape) for grad in grads]

        self.u[key] = [self._configuration.beta2 * u + ((1 - self._configuration.beta2) * (grad * grad)) for u, grad in zip(self.u[key], grads)]
        u_corrected = [u / (1 - self._configuration.beta2) for u in self.u[key]]

        deltas = [-self._configuration.alpha * m_head / np.sqrt(self.Lambda + u_head) for m_head, u_head in zip(m_corrected, u_corrected)]
        self.last_deltas[key] = deltas
        return deltas


class OptimizerFactory:
    @staticmethod
    def instance(configuration: OptimizerConfiguration):
        if configuration.name == OptimizerEnum.GDOptimizer:
            return GDOptimizer(configuration)
        if configuration.name == OptimizerEnum.AdaGradOptimizer:
            return AdaGradOptimizer(configuration)
        if configuration.name == OptimizerEnum.RMSPropOptimizer:
            return RMSPropOptimizer(configuration)
        if configuration.name == OptimizerEnum.MomentumOptimizer:
            return MomentumOptimizer(configuration)
        if configuration.name == OptimizerEnum.ADAMOptimizer:
            return AdamOptimizer(configuration)

        return None
