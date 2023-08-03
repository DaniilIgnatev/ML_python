#%%
import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

random.seed(1)
np.random.seed(1)

dimensions_size = 32

root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
train_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_025_0_360_15_15', os.path.join(root_path, 'datasets'),
                                                     dimensions_size, 0.5, 3.5, 0.25, 0, 360, 15, 15, False)
test_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_0375_0_360_20_0', os.path.join(root_path, 'datasets'),
                                                    dimensions_size, 0.5, 3.5, 0.375, 0, 360, 20, 0, False)
optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 0.5, 0, 0, True)
n_config = NeuralNetworkConfiguration(sizes=[dimensions_size**2, 700, 1], lambda_l1=0.01, lambda_l2=0.01,
                                      optimizer_configuration=optimizer_configuration, output=True)
c_config = ClassifierConfiguration(f'NN_{dimensions_size}_700_1_lr=05', os.path.join(root_path, 'neural_networks'),
                                   FiguresEnum.TRIANGLE, dimensions_size, train_dataset_config, test_dataset_config, n_config)
c = Classifier(c_config)

if not c.is_trained():
    c.train_classifier()


#%%
import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

random.seed(1)
np.random.seed(1)

dimensions_size = 32

root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
train_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_025_0_360_15_15', os.path.join(root_path, 'datasets'),
                                                     dimensions_size, 0.5, 3.5, 0.25, 0, 360, 15, 15, False)
test_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_0375_0_360_20_0', os.path.join(root_path, 'datasets'),
                                                    dimensions_size, 0.5, 3.5, 0.375, 0, 360, 20, 0, False)
optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 1, 0, 0, True)
n_config = NeuralNetworkConfiguration(sizes=[dimensions_size**2, 700, 1], lambda_l1=0.01, lambda_l2=0.01,
                                      optimizer_configuration=optimizer_configuration, output=True)
c_config = ClassifierConfiguration(f'NN_{dimensions_size}_700_1_lr=1', os.path.join(root_path, 'neural_networks'),
                                   FiguresEnum.TRIANGLE, dimensions_size, train_dataset_config, test_dataset_config, n_config)
c = Classifier(c_config)

if not c.is_trained():
    c.train_classifier()


#%%
import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

random.seed(1)
np.random.seed(1)

dimensions_size = 32

root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
train_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_025_0_360_15_15', os.path.join(root_path, 'datasets'),
                                                     dimensions_size, 0.5, 3.5, 0.25, 0, 360, 15, 15, False)
test_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_0375_0_360_20_0', os.path.join(root_path, 'datasets'),
                                                    dimensions_size, 0.5, 3.5, 0.375, 0, 360, 20, 0, False)
optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 0.5, 0, 0, True)
n_config = NeuralNetworkConfiguration(sizes=[dimensions_size**2, 200, 10, 1], lambda_l1=0.01, lambda_l2=0.01,
                                      optimizer_configuration=optimizer_configuration, output=True)
c_config = ClassifierConfiguration(f'NN_{dimensions_size}_200_10_lr=05', os.path.join(root_path, 'neural_networks'),
                                   FiguresEnum.TRIANGLE, dimensions_size, train_dataset_config, test_dataset_config, n_config)
c = Classifier(c_config)

if not c.is_trained():
    c.train_classifier()


#%%
import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

random.seed(1)
np.random.seed(1)

dimensions_size = 32

root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
train_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_025_0_360_15_15', os.path.join(root_path, 'datasets'),
                                                     dimensions_size, 0.5, 3.5, 0.25, 0, 360, 15, 15, False)
test_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_0375_0_360_20_0', os.path.join(root_path, 'datasets'),
                                                    dimensions_size, 0.5, 3.5, 0.375, 0, 360, 20, 0, False)
optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 1, 0, 0, True)
n_config = NeuralNetworkConfiguration(sizes=[dimensions_size**2, 200, 10, 1], lambda_l1=0.01, lambda_l2=0.01,
                                      optimizer_configuration=optimizer_configuration, output=True)
c_config = ClassifierConfiguration(f'NN_{dimensions_size}_200_10_lr=1', os.path.join(root_path, 'neural_networks'),
                                   FiguresEnum.TRIANGLE, dimensions_size, train_dataset_config, test_dataset_config, n_config)
c = Classifier(c_config)

if not c.is_trained():
    c.train_classifier()


#%%
import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

random.seed(1)
np.random.seed(1)

dimensions_size = 32

root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
train_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_025_0_360_15_15', os.path.join(root_path, 'datasets'),
                                                     dimensions_size, 0.5, 3.5, 0.25, 0, 360, 15, 15, False)
test_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_0375_0_360_20_0', os.path.join(root_path, 'datasets'),
                                                    dimensions_size, 0.5, 3.5, 0.375, 0, 360, 20, 0, False)
optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 1, 0, 0, True)
n_config = NeuralNetworkConfiguration(sizes=[dimensions_size**2, 200, 10, 1], lambda_l1=0.01, lambda_l2=0.01,
                                      optimizer_configuration=optimizer_configuration, output=True)
c_config = ClassifierConfiguration(f'NN_{dimensions_size}_200_10_lr=1', os.path.join(root_path, 'neural_networks'),
                                   FiguresEnum.TRIANGLE, dimensions_size, train_dataset_config, test_dataset_config, n_config)
c = Classifier(c_config)

if not c.is_trained():
    c.train_classifier()


#%%
import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

random.seed(1)
np.random.seed(1)

dimensions_size = 32

root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
train_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_025_0_360_15_15', os.path.join(root_path, 'datasets'),
                                                     dimensions_size, 0.5, 3.5, 0.25, 0, 360, 15, 15, False)
test_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_0375_0_360_20_0', os.path.join(root_path, 'datasets'),
                                                    dimensions_size, 0.5, 3.5, 0.375, 0, 360, 20, 0, False)
optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 1, 0, 0, True)
n_config = NeuralNetworkConfiguration(sizes=[dimensions_size**2, 150, 10, 1], lambda_l1=0.01, lambda_l2=0.01,
                                      optimizer_configuration=optimizer_configuration, output=True)
c_config = ClassifierConfiguration(f'NN_{dimensions_size}_200_10_lr=1', os.path.join(root_path, 'neural_networks'),
                                   FiguresEnum.TRIANGLE, dimensions_size, train_dataset_config, test_dataset_config, n_config)
c = Classifier(c_config)

if not c.is_trained():
    c.train_classifier()


#%%
import os
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum

random.seed(1)
np.random.seed(1)

dimensions_size = 32

root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
train_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_025_0_360_15_15', os.path.join(root_path, 'datasets'),
                                                     dimensions_size, 0.5, 3.5, 0.25, 0, 360, 15, 15, False)
test_dataset_config = DatasetGeneratorConfiguration(f'{dimensions_size}_05_35_0375_0_360_20_0', os.path.join(root_path, 'datasets'),
                                                    dimensions_size, 0.5, 3.5, 0.375, 0, 360, 20, 0, False)
optimizer_configuration = OptimizerConfiguration(OptimizerEnum.GDOptimizer, 1, 0, 0, True)
n_config = NeuralNetworkConfiguration(sizes=[dimensions_size**2, 100, 10, 1], lambda_l1=0.01, lambda_l2=0.01,
                                      optimizer_configuration=optimizer_configuration, output=True)
c_config = ClassifierConfiguration(f'NN_{dimensions_size}_200_10_lr=1', os.path.join(root_path, 'neural_networks'),
                                   FiguresEnum.TRIANGLE, dimensions_size, train_dataset_config, test_dataset_config, n_config)
c = Classifier(c_config)

if not c.is_trained():
    c.train_classifier()