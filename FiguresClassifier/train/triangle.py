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


triangle_classifier = c_factory.get_classifier(FiguresEnum.TRIANGLE,
                                               min_scale=min_scale, max_scale=max_scale, scale_precision=scale_precision, angle_precision=angle_precision,
                                               distortion_percentage=distortion_medium, save_plots=save_plots,
                                               optimizer_alpha=optimizer_alpha, optimizer_beta=optimizer_beta,
                                               nn_h1=nn_h1, nn_h2=nn_h2, nn_l1=0.3, nn_l2=0.01
                                               )

if __name__ == "__main__":
    triangle_classifier.train_classifier()

#%%