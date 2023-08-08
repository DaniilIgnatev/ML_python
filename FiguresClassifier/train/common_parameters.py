import os
import sys
from FiguresClassifier.Classifier.classifier_factory import ClassifierFactory

dimensions_size = 32

if getattr(sys, 'frozen', False):
    root_path = os.path.dirname(sys.executable)
else:
    root_path = os.path.abspath(os.getcwd())

if root_path.endswith('train'):
    root_path = os.path.dirname(root_path)

if 'FiguresClassifier' not in root_path:
    root_path = os.path.join(root_path, 'FiguresClassifier')

root_path = os.path.join(root_path, 'save')
print(f'root_path: {root_path}')

c_factory = ClassifierFactory(dimensions_size, root_path)

# mini config
# min_scale = 0.5
# max_scale = 3.0
# scale_precision = 1.0
# angle_precision = 45
# distortion_low = 2
# distortion_medium = 5
# distortion_high = 10
# save_plots = False
#
# optimizer_alpha = 0.01
# optimizer_beta = 0.9
#
# nn_h1 = 100
# nn_h2 = 10
# nn_l1 = 0.001
# nn_l2 = 0.1

# main config
min_scale = 0.5
max_scale = 3.0
scale_precision = 0.25
angle_precision = 25
distortion_low = 5
distortion_medium = 10
distortion_high = 25
save_plots = False

optimizer_alpha = 0.01
optimizer_beta = 0.9

nn_h1 = 200
nn_h2 = 10
