from FiguresClassifier.Figures.generator import FiguresEnum
from FiguresClassifier.train.common_parameters import c_factory

from FiguresClassifier.train.common_parameters import min_scale
from FiguresClassifier.train.common_parameters import max_scale
from FiguresClassifier.train.common_parameters import scale_precision
from FiguresClassifier.train.common_parameters import angle_precision
from FiguresClassifier.train.common_parameters import distortion_low
from FiguresClassifier.train.common_parameters import distortion_medium
from FiguresClassifier.train.common_parameters import distortion_high
from FiguresClassifier.train.common_parameters import save_plots

from FiguresClassifier.train.common_parameters import optimizer_alpha
from FiguresClassifier.train.common_parameters import optimizer_beta

from FiguresClassifier.train.common_parameters import nn_h1
from FiguresClassifier.train.common_parameters import nn_h2


line_classifier = c_factory.get_classifier(FiguresEnum.LINE,
                                           min_scale=1, max_scale=1, scale_precision=1, angle_precision=20,
                                           distortion_percentage=distortion_medium, save_plots=False,
                                           optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                           nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=0.0, nn_l2=0.01
                                           )

if __name__ == "__main__":
    line_classifier.train_classifier()

#%%
