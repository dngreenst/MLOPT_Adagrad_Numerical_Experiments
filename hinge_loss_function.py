import unittest

import numpy as np


class HingeLossFunction:
    def __init__(self, label: float, features: np.array):
        self._label = label
        self._features = features
        self._scale = 1.0

    def loss(self, x: np.array) -> float:
        return self._scale * max(0.0, 1 - self._label * np.inner(self._features, x))

    def gradient(self, x):

        if self.loss(x) > 0:
            return self._scale * (-self._label * self._features)

        return np.zeros_like(self._features)


class test_hinge_loss_function(unittest.TestCase):

    def test_loss(self):

        features = np.array([0.6, 0.3, 8, 0.85])
        label = -1
        x = np.array([0.3, 0.7, 0.2, 0.1])

        hinge = HingeLossFunction(label=label, features=features)
        print(hinge.loss(x))

    def test_gradient(self):
        features = np.array([0.6, 0.3, 8, 0.85])
        label = -1
        x = np.array([0.3, 0.7, 0.2, 0.1])

        hinge = HingeLossFunction(label=label, features=features)

        print(hinge.gradient(x))
