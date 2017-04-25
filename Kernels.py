import numpy as np


class Linear:
    def __call__(self, a, b):
        x = np.array(a)
        y = np.array(b)
        y = np.transpose(y)
        return np.dot(x, y)

    @staticmethod
    def get_name():
        return "Linear"


class Polynomial:
    def __call__(self, a, b, p=2):
        self.p = p
        x = np.array(a)
        y = np.array(b)
        y = np.transpose(y)
        return (1 + np.dot(x, y)) ** p

    @staticmethod
    def get_name():
        return "Quadratic"


class Gaussian:
    def __call__(self, a, b, sigma=5.0):
        x = np.array(a)
        y = np.array(b)
        y = np.transpose(y)
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def get_name():
        return "Gaussian"
