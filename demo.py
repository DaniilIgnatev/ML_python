import os
import sys
import random
import numpy as np
from Figures.classifier import ClassifierConfiguration
from Figures.classifier import Classifier
from Figures.classifier_factory import ClassifierFactory
from Figures.train_data import DatasetGeneratorConfiguration
from Figures.figures import FiguresEnum
from NN.neural_network import NeuralNetworkConfiguration
from NN.optimizer import OptimizerConfiguration
from NN.optimizer import OptimizerEnum
from Figures.figures import FigureData

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

random.seed(1)
np.random.seed(1)

dimensions_size = 32

if getattr(sys, 'frozen', False):
    root_path = os.path.dirname(sys.executable)
else:
    root_path = os.path.abspath(os.getcwd())
# root_path = 'C:\\Users\\Daniil\\Desktop\\FiguresClassifier'
print(root_path)

c_factory = ClassifierFactory(dimensions_size, root_path)

# min_scale=1.0
# max_scale=3.0
# scale_precision=0.5
# angle_precision=30
# distortion_percentage=10
# save_plots=False
#
# optimizer_alpha=0.01
# optimizer_beta=0.9
#
# nn_h1=50
# nn_h2=10
# nn_l1=0.0
# nn_l2=10

# min_scale=1.0
# max_scale=3.0
# scale_precision=0.25
# angle_precision=15
# distortion_percentage=10
# save_plots=False
#
# optimizer_alpha=0.01
# optimizer_beta=0.9
#
# nn_h1=100
# nn_h2=10
# nn_l1=0.01
# nn_l2=10

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

noise_classifier = c_factory.get_classifier(FiguresEnum.NOISE,
                                            min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                            distortion_percentage=distortion_percentage, save_plots=save_plots,
                                            optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                            nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                            )
if not noise_classifier.is_trained():
    print(f'training the noise_classifier')
    noise_classifier.train_classifier()
else:
    print('loading the noise_classifier')

# line_classifier = c_factory.get_classifier(FiguresEnum.LINE)
# if not line_classifier.is_trained():
#     print(f'training the line_classifier')
#     line_classifier.train_classifier()
# else:
#     print('loading the line_classifier')

triangle_classifier = c_factory.get_classifier(FiguresEnum.TRIANGLE,
                                               min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                               distortion_percentage=distortion_percentage, save_plots=save_plots,
                                               optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                               nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                               )
if not triangle_classifier.is_trained():
    print(f'training the triangle_classifier')
    triangle_classifier.train_classifier()
else:
    print('loading the triangle_classifier')

rectangle_classifier = c_factory.get_classifier(FiguresEnum.RECTANGLE,
                                                min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                                distortion_percentage=distortion_percentage, save_plots=save_plots,
                                                optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                                nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                                )
if not rectangle_classifier.is_trained():
    print(f'training the rectangle_classifier')
    rectangle_classifier.train_classifier()
else:
    print('loading the rectangle_classifier')

ellipse_classifier = c_factory.get_classifier(FiguresEnum.ELLIPSE,
                                              min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                              distortion_percentage=distortion_percentage, save_plots=save_plots,
                                              optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                              nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=nn_l1, nn_l2=nn_l2
                                              )
if not ellipse_classifier.is_trained():
    print(f'training the ellipse_classifier')
    ellipse_classifier.train_classifier()
else:
    print('loading the ellipse_classifier')

print("Noise accuracy: ", noise_classifier.training_log.max_accuracy)
# print("Line accuracy: ", line_classifier.training_log.max_accuracy)
print("Triangle accuracy: ", triangle_classifier.training_log.max_accuracy)
print("Rectangle accuracy: ", rectangle_classifier.training_log.max_accuracy)
print("Ellipse accuracy: ", ellipse_classifier.training_log.max_accuracy)

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_xlim(0, dimensions_size * 4)
ax.set_ylim(0, dimensions_size * 4)
ax.invert_yaxis()

# Create a button
process_button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
process_button = Button(process_button_ax, 'Process')

clear_button_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
clear_button = Button(clear_button_ax, 'Clear')

# Create a textbox
text_box_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
text_box = TextBox(text_box_ax, '', initial="")

# Draw empty plot
line, = ax.plot([], [], color='black', marker='o', markersize=2)

xdata = []
ydata = []
is_pressed = False


# Mouse button press event
def on_press(event):
    # print('on_press')
    global is_pressed
    is_pressed = True


# Mouse motion event
def on_motion(event):
    global is_pressed
    if is_pressed:
        # print('on_motion')
        if event.xdata:
            if event.xdata > 1 and event.ydata > 1:
                xdata.append(event.xdata)
                ydata.append(event.ydata)
                line.set_data(xdata, ydata)
                fig.canvas.draw()


# Mouse button release event
def on_release(event):
    # print('on_release')
    global is_pressed
    is_pressed = False


def process_data(event):
    print('process_data')

    global is_pressed
    is_pressed = False

    global dimensions_size
    global xdata, ydata

    x = np.array(xdata)
    mask_x = x > 1
    x = x[mask_x]

    y = np.array(ydata)
    mask_y = y > 1
    y = y[mask_y]

    data = FigureData(FiguresEnum.NOISE, dimensions_size, x, y)
    data.shift_to_zero()
    data.scale_to_fit()
    x = data.x
    y = data.y
    x_interpolated = np.array([])
    y_interpolated = np.array([])

    for i in range(len(x)):
        if i != len(x) - 1:
            x_n = (x[i] + x[i + 1]) / 2
            y_n = (y[i] + y[i + 1]) / 2
            x_interpolated = np.append(x_interpolated, x_n)
            y_interpolated = np.append(y_interpolated, y_n)

            x_n2 = (x[i] + x_n) / 2
            y_n2 = (y[i] + y_n) / 2
            x_interpolated = np.append(x_interpolated, x_n2)
            y_interpolated = np.append(y_interpolated, y_n2)

            x_n3 = (x[i + 1] + x_n) / 2
            y_n3 = (y[i + 1] + y_n) / 2
            x_interpolated = np.append(x_interpolated, x_n3)
            y_interpolated = np.append(y_interpolated, y_n3)

            x_n4 = (x[i] + x_n2) / 2
            y_n4 = (y[i] + y_n2) / 2
            x_interpolated = np.append(x_interpolated, x_n4)
            y_interpolated = np.append(y_interpolated, y_n4)

            x_n5 = (x_n2 + x_n3) / 2
            y_n5 = (y_n2 + y_n3) / 2
            x_interpolated = np.append(x_interpolated, x_n5)
            y_interpolated = np.append(y_interpolated, y_n5)

    x = np.append(x, x_interpolated)
    y = np.append(y, y_interpolated)
    data.set_xy(x, y)

    data.clip()
    data.filter()

    if len(data.points) > 1:
        noise_p = noise_classifier.classify(x, y)
        line_p = 0#line_classifier.classify(x, y)
        triangle_p = triangle_classifier.classify(x, y)
        rectangle_p = rectangle_classifier.classify(x, y)
        ellipse_p = ellipse_classifier.classify(x, y)

        print(f'n: {noise_p}; l:{line_p}; t:{triangle_p}; r:{rectangle_p}; e:{ellipse_p}')

        P = [noise_p, line_p, triangle_p, rectangle_p, ellipse_p]
        soft_max_p = softmax(P)
        print(f'soft_max: {soft_max_p}')

        labels = ['Noise', 'Line', 'Triangle', 'Rectangle', 'Ellipse']
        label_index = np.argmax(soft_max_p)

        text_box.set_val(f'{labels[label_index]}')

        data.plot()


def clear_plot(event):
    global is_pressed
    is_pressed = False

    global xdata, ydata
    xdata = []
    ydata = []
    line.set_data(xdata, ydata)
    text_box.set_val('')
    fig.canvas.draw()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Connect the events to the functions
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

process_button.on_clicked(process_data)
clear_button.on_clicked(clear_plot)

plt.show()



#%%
