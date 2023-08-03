import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3
import autograd.numpy as np
from autograd import grad
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import seaborn as sns
import random

from NN.neural_network import NeuralNetwork
from NN.neural_network import NeuralNetworkConfiguration

from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum
from NN.optimizer import OptimizerFactory


if __name__ == "__main__":
    # Run this code to see if your code is correct
    random.seed(42)
    np.random.seed(2)
    optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 1, 0, 0, True)
    configuration = NeuralNetworkConfiguration(sizes=[3, 2, 2, 1], lambda_l1=0.00001, lambda_l2=0.00001, optimizer_configuration=optimizer_configuration, output=True)
    network = NeuralNetwork(configuration)
    np.random.seed(None)
    x1 = np.array([[1], [2], [3]])
    r = network.forward_pass(x1, return_lists=True)
    print(r)
    r_test = (
        [
            np.array([[1], [2], [3]]),
            np.array([[0.01818856],[0.21791265]]),
            np.array([[0.11051885], [0.85035715]]),
            np.array([[0.13289621]])
        ],
        [
            None,
            np.array([[0.01785773],[0.17042673]]),
            np.array([[0.09830444],[0.12724987]]),
            np.array([[0.11523481]])
        ]
    )


    # %%
    # Run this code to see if your code is correct
    y1 = np.array([[1]])
    a1, a_I1 = network.forward_pass(x1, return_lists=True)
    r = network.backward_pass(a1, a_I1, y1)
    print(r)

    # ([
    #   array([[0.00203797, 0.00407595, 0.00611392],[0.00055368, 0.00110735, 0.00166103]]),
    #   array([[9.24283189e-04, 1.10736109e-02], [3.83930752e-05, 4.59978048e-04]]),
    #   array([[-0.09583132, -0.73734791]])
    #  ],
    #  [
    #   array([[0.00203797],[0.00055368]]),
    #   array([[0.05081674],[0.00211084]]),
    #   array([[-0.86710379]])
    #  ])
    #  [0.00211084]]), array([[-0.86710379]])])


    # %%
    def generate_non_lin_sep_data(n, rs=None, plot=False, type='moons'):
        global X_n, y_n
        if type == 'circless':
            X_n, y_n = datasets.make_circles(n_samples=n, noise=0.001, random_state=rs)
        if type == 'moons':
            X_n, y_n = datasets.make_moons(n_samples=n, noise=0.1, random_state=rs)
        if type == 'class':
            X_n, y_n = datasets.make_classification(n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                                    class_sep=2.0, random_state=rs)
        df = pd.DataFrame(data=np.concatenate((X_n, y_n.reshape((y_n.shape[0], 1))), axis=1))
        # rename columns
        num_cols = len(list(df)) - 1
        rng = range(1, num_cols + 1)
        new_cols = ['x' + str(i) for i in rng] + ['y']
        df.columns = new_cols
        if plot:
            sns.set(style="ticks")
            sns.pairplot(df, hue="y")
        return df


    # %%
    data = generate_non_lin_sep_data(1000, 2, True)
    data = data.to_numpy()
    random.seed(1)
    np.random.seed(1)
    test_index = np.random.choice([True, False], len(data), replace=True, p=[0.25, 0.75])
    test = data[test_index]
    train = data[np.logical_not(test_index)]
    train = [(d[:2][:, np.newaxis], np.array([[d[-1]]])) for d in train]
    test = [(d[:2][:, np.newaxis], d[-1]) for d in test]


    # %%
    random.seed(1)
    np.random.seed(1)
    optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 10, 0, 0, True)
    configuration = NeuralNetworkConfiguration(sizes=[2, 20, 1], lambda_l1=0.001, lambda_l2=0.001, optimizer_configuration=optimizer_configuration, output=True)
    network = NeuralNetwork(configuration)
    print(f'Samples successfully classified before training: {network.evaluate(test)} of {len(test)}')
    network.train(training_data=train, epochs=10, mini_batch_size=20, test_data=test,
                  random_shuffle=True)
    print(f'Samples successfully classified after training: {network.evaluate(test)} of {len(test)}')

