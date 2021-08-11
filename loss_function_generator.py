import copy

import numpy as np
from numpy.random import default_rng

from hinge_loss_function import HingeLossFunction


def generate_loss_functions(labels: np.array, features: np.array):

    loss_functions = []

    const = 10
    alpha = 1.1

    samples_num = len(labels)
    features_num = len(features[0])

    feature_probabilities_list = [0.0]
    feature_probabilities_list.extend([min(1.0, const * np.power(1/float(i), alpha)) for i in range(1, features_num)])
    feature_probabilities = np.array(feature_probabilities_list)
    feature_probabilities = np.minimum(feature_probabilities, 0.7)
    rng = default_rng()

    for i in range(samples_num):
        tmp_features = copy.deepcopy(features[i])

        for feature_idx in range(features_num):
            if rng.uniform(low=0.0, high=1.0) < feature_probabilities[feature_idx]:
                tmp_features[feature_idx] = 0.0

        loss_functions.append(HingeLossFunction(label=labels[i], features=tmp_features))

    return loss_functions
