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
from Figures.classifier_factory import ClassifierFactory


dimensions_size = 32
root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
c_factory = ClassifierFactory(dimensions_size, root_path)

min_scale=1.0
max_scale=3.0
scale_precision=0.25
angle_precision=15
distortion_percentage=5
save_plots=False

optimizer_alpha=0.001
optimizer_beta=0.9

nn_h1=100
nn_h2=10
nn_l1=0.001
nn_l2=0.1

rectangle_classifier = c_factory.get_classifier(FiguresEnum.RECTANGLE,
                                        min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                        distortion_percentage=distortion_percentage, save_plots=save_plots,
                                        optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                        nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                        )
rectangle_classifier.train_classifier()


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
from Figures.classifier_factory import ClassifierFactory


dimensions_size = 32
root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
c_factory = ClassifierFactory(dimensions_size, root_path)

min_scale=1.0
max_scale=3.0
scale_precision=0.25
angle_precision=15
distortion_percentage=5
save_plots=False

optimizer_alpha=0.001
optimizer_beta=0.9

nn_h1=100
nn_h2=10
nn_l1=0.01
nn_l2=1.0

rectangle_classifier = c_factory.get_classifier(FiguresEnum.RECTANGLE,
                                                min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                                distortion_percentage=distortion_percentage, save_plots=save_plots,
                                                optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                                nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                                )
rectangle_classifier.train_classifier()




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
from Figures.classifier_factory import ClassifierFactory


dimensions_size = 32
root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
c_factory = ClassifierFactory(dimensions_size, root_path)

min_scale=1.0
max_scale=3.0
scale_precision=0.25
angle_precision=15
distortion_percentage=5
save_plots=False

optimizer_alpha=0.001
optimizer_beta=0.9

nn_h1=100
nn_h2=10
nn_l1=0.1
nn_l2=10.0

rectangle_classifier = c_factory.get_classifier(FiguresEnum.RECTANGLE,
                                                min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                                distortion_percentage=distortion_percentage, save_plots=save_plots,
                                                optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                                nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                                )
rectangle_classifier.train_classifier()




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
from Figures.classifier_factory import ClassifierFactory


dimensions_size = 32
root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
c_factory = ClassifierFactory(dimensions_size, root_path)

min_scale=1.0
max_scale=3.0
scale_precision=0.25
angle_precision=15
distortion_percentage=5
save_plots=False

optimizer_alpha=0.001
optimizer_beta=0.9

nn_h1=100
nn_h2=10
nn_l1=1.0
nn_l2=100.0

rectangle_classifier = c_factory.get_classifier(FiguresEnum.RECTANGLE,
                                                min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                                distortion_percentage=distortion_percentage, save_plots=save_plots,
                                                optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                                nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                                )
rectangle_classifier.train_classifier()
