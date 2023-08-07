import os
from FiguresClassifier.Classifier.classifier_factory import ClassifierFactory

dimensions_size = 32

root_path = os.path.abspath(os.getcwd())
root_path = os.path.dirname(root_path)
root_path = os.path.join(root_path, 'save')
print(f'root_path: {root_path}')

c_factory = ClassifierFactory(dimensions_size, root_path)

min_scale = 0.5
max_scale = 3.0
scale_precision = 0.25
angle_precision = 15
distortion_small = 2
distortion_medium = 5
distortion_high = 10
save_plots = False

optimizer_alpha = 0.01
optimizer_beta = 0.9

nn_h1 = 100
nn_h2 = 10
