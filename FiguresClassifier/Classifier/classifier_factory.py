import os

from FiguresClassifier.Classifier.classifier import Classifier
from FiguresClassifier.Classifier.classifier import ClassifierConfiguration
from FiguresClassifier.Dataset.generator import DatasetGeneratorConfiguration
from FiguresClassifier.Figures.generator import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum


class ClassifierFactory:
    def __init__(self, dimensions_size, root_path):
        self.dimensions_size = dimensions_size
        self.root_path = root_path

    def get_classifier(self, target_name: FiguresEnum,
                       min_scale=0.5, max_scale=3.5, scale_precision=0.5, angle_precision=30, distortion_percentage=5,
                       save_plots=False,
                       optimizer_alpha=0.01, optimizer_beta=0.9,
                       nn_h1=100, nn_h2=10, nn_l1=0.01, nn_l2=0.01):
        train_min_scale = min_scale
        train_max_scale = max_scale
        train_scale_precision = scale_precision
        train_min_angle = 0
        train_max_angle = 360
        train_angle_precision = angle_precision
        train_distortion_percentage = distortion_percentage
        train_save_plots = save_plots
        self.train_dataset_config = DatasetGeneratorConfiguration(f'{self.dimensions_size}_{train_min_scale}_{train_max_scale}_{train_scale_precision}_{train_min_angle}_{train_max_angle}_{train_angle_precision}_{train_distortion_percentage}',
                                                                  os.path.join(self.root_path, 'datasets'),
                                                                  self.dimensions_size, train_min_scale, train_max_scale, train_scale_precision, train_min_angle, train_max_angle, train_angle_precision, train_distortion_percentage, train_save_plots)

        test_min_scale = train_min_scale
        test_max_scale = train_max_scale
        test_scale_precision = train_scale_precision * 1.5
        test_min_angle = train_min_angle
        test_max_angle = train_max_angle
        test_angle_precision = train_angle_precision * 1.5
        test_distortion_percentage = distortion_percentage
        test_save_plots = save_plots
        self.test_dataset_config = DatasetGeneratorConfiguration(f'{self.dimensions_size}_{test_min_scale}_{test_max_scale}_{test_scale_precision}_{test_min_angle}_{test_max_angle}_{test_angle_precision}_{test_distortion_percentage}',
                                                                 os.path.join(self.root_path, 'datasets'),
                                                                 self.dimensions_size, test_min_scale, test_max_scale, test_scale_precision, test_min_angle, test_max_angle, test_angle_precision, test_distortion_percentage, test_save_plots)

        optimizer_name = OptimizerEnum.RMSPropOptimizer
        optimizer_alpha = optimizer_alpha
        optimizer_beta = optimizer_beta
        optimizer_record = True

        self.optimizer_configuration = OptimizerConfiguration(optimizer_name, alpha=optimizer_alpha, beta1=optimizer_beta, record=optimizer_record)

        nn_h1 = nn_h1
        nn_h2 = nn_h2
        nn_l1 = nn_l1
        nn_l2 = nn_l2
        nn_mini_batch_size = 64
        nn_output = True

        self.nn_config = NeuralNetworkConfiguration(sizes=[self.dimensions_size**2, nn_h1, nn_h2, 1], lambda_l1=nn_l1, lambda_l2=nn_l2,
                                                    optimizer_configuration=self.optimizer_configuration, mini_batch_size=nn_mini_batch_size, output=nn_output)

        c_config = ClassifierConfiguration(f'{self.dimensions_size}_{target_name.name}_{self.nn_config.sizes[1]}_{self.nn_config.sizes[2]}_{self.nn_config.lambda_l1}_{self.nn_config.lambda_l2}_{self.nn_config.mini_batch_size}_{self.optimizer_configuration.name}_{self.optimizer_configuration.alpha}_{self.optimizer_configuration.beta1}',
                                           os.path.join(self.root_path, 'parameters'),
                                           target_name, self.dimensions_size, self.train_dataset_config, self.test_dataset_config, self.nn_config)

        return Classifier(c_config)
