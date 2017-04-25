from Kernels import *
import numpy as np


class SvmProblem:
    """
    All information about a problem that can be solved by:-
        - SVM
        - SVM+
        - SVMd+
        - MT / SVMd+:simp
        - KT

    Attributes
    ----------
    C : float
        The regularization parameter to use for this problem
    gamma : float
        Parameter controlling that priority between X and X* (if applicable)
    delta : float
        Parameter controlling the strictness of `classification' in X* (if applicable)
    xkernel: Kernel
        Kernel used in X
    xskernel: Kernel
        Kernel used in X* (if applicable)
    sigma: float
        Standard deviation in distance between datapoints in space where Gaussian kernel is applied
    X : numpy.array
        Non-privileged data-points
    Xstar : numpy.array
        Privileged data-points
    Y : numpy.array
        Class labels
    num : int
        Number of data-points
    dimensions: int
        Number of features in X
    xi_xj : numpy.ndarray
        K(x,x) Matrix of kernel distances between each data-point in X
    xstari_xstarj : numpy.ndarray
        K*(x*,x*) Matrix of kernel distances between each data-point in X*
    yi_yj : numpy.array
        Array of Yi * Yj

    """
    def __init__(self, x, xstar, y, c=1.0, gamma=1.0, delta=1000.0, xk=Linear(), xsk=Linear()):
        self.C = c
        self.gamma = gamma
        self.delta = delta
        self.xkernel = xk
        self.xskernel = xsk
        self.sigma = -99

        if isinstance(x, np.ndarray):
            self.X = x
        else:
            self.X = np.array(x)
        if isinstance(xstar, np.ndarray):
            self.Xstar = xstar
        else:
            self.Xstar = np.array(xstar)
        if isinstance(y, np.ndarray):
            self.Y = y
            self.Y = np.asarray(self.Y).reshape(-1)
        else:
            self.Y = np.array(y)
            self.Y = np.asarray(self.Y).reshape(-1)

        self.num = len(self.X)
        self.dimensions = len(self.X[0])
        self.xi_xj = self.gram_matrix(self.X, self.X, self.xkernel)
        self.xstari_xstarj = self.gram_matrix(self.Xstar, self.Xstar, self.xskernel)
        self.yi_yj = self.gram_matrix(self.Y, self.Y, Linear())

    def gram_matrix(self, x1, x2, kern):
        """
        Return matrix comparing x1 to x2 using kern

        Parameters
        ----------
        x1 : numpy.ndarray
            Feature arrays representing data-points / class labels
        x2 : numpy.ndarray
            Feature arrays representing data-points / class labels
        kern : Kernel
            Kernel distance measure to use

        Returns
        -------
        numpy.ndarray
            Matrix of kernel distances between each pairing of inputs

        """
        if isinstance(kern, Gaussian):
            if self.sigma == -99:
                sqd = np.zeros((len(x1), len(x1)))
                for i in range(len(x1)):
                    for j in range(len(x1)):
                        sqd[i, j] = np.linalg.norm(x1[i] - x2[j])
                self.sigma = np.median(sqd)
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j], sigma=self.sigma)
            return k
        else:
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j])
            return k


class SvmProblemTuple:
    """
    All information about a problem that can be solved by:-
        - SVM
        - SVM+
        - SVMd+
        - MT / SVMd+:simp
        - KT

    Attributes
    ----------
    C : float
        The regularization parameter to use for this problem
    gamma : float
        Parameter controlling that priority between X and X* (if applicable)
    delta : float
        Parameter controlling the strictness of `classification' in X* (if applicable)
    xkernel: Kernel
        Kernel used in X
    xskernel: Kernel
        Kernel used in X* (if applicable)
    sigma: float
        Standard deviation in distance between datapoints in space where Gaussian kernel is applied
    X : numpy.array
        Non-privileged data-points
    Xstar : numpy.array
        Privileged data-points
    Y : numpy.array
        Class labels
    num : int
        Number of data-points
    dimensions: int
        Number of features in X
    xi_xj : numpy.ndarray
        K(x,x) Matrix of kernel distances between each data-point in X
    xstari_xstarj : numpy.ndarray
        K*(x*,x*) Matrix of kernel distances between each data-point in X*
    yi_yj : numpy.array
        Array of Yi * Yj

    """
    def __init__(self, prob_tuple):
        self.C = prob_tuple[5]
        self.sigma = -99
        if len(prob_tuple) == 9:  # SVM
            self.gamma = 1
            self.delta = 1
            self.xkernel = prob_tuple[6]
            self.xskernel = Linear()
        elif len(prob_tuple) == 11:  # SVM+
            self.gamma = prob_tuple[6]
            self.delta = 1
            self.xkernel = prob_tuple[7]
            self.xskernel = prob_tuple[8]
        elif len(prob_tuple) == 12:  # SVMd+ - sa
            self.gamma = prob_tuple[6]
            self.delta = prob_tuple[7]
            self.xkernel = prob_tuple[8]
            self.xskernel = prob_tuple[9]
        elif len(prob_tuple) == 13:  # SVMd+
            self.gamma = prob_tuple[7]
            self.delta = prob_tuple[6]
            self.xkernel = prob_tuple[8]
            self.xskernel = prob_tuple[9]
        elif len(prob_tuple) == 15:  # KT
            self.gamma = prob_tuple[6]
            self.delta = 1000
            self.xkernel = prob_tuple[7]
            self.xskernel = prob_tuple[8]
        else:
            print("poorly formed problem")

        if isinstance(prob_tuple[0], np.ndarray):
            self.X = prob_tuple[0]
        else:
            self.X = np.array(prob_tuple[0])
        if isinstance(prob_tuple[1], np.ndarray):
            self.Xstar = prob_tuple[1]
        else:
            self.Xstar = np.array(prob_tuple[1])
        if isinstance(prob_tuple[2], np.ndarray):
            self.Y = prob_tuple[2]
            self.Y = np.asarray(self.Y).reshape(-1)
        else:
            self.Y = np.array(prob_tuple[2])
            self.Y = np.asarray(self.Y).reshape(-1)

        self.num = len(self.X)
        self.dimensions = len(self.X[0])
        self.xi_xj = self.gram_matrix(self.X, self.X, self.xkernel)
        self.xstari_xstarj = self.gram_matrix(self.Xstar, self.Xstar, self.xskernel)
        self.yi_yj = self.gram_matrix(self.Y, self.Y, Linear())

    def gram_matrix(self, x1, x2, kern):
        """
        Return matrix comparing x1 to x2 using kern

        Parameters
        ----------
        x1 : numpy.ndarray
            Feature arrays representing data-points / class labels
        x2 : numpy.ndarray
            Feature arrays representing data-points / class labels
        kern : Kernel
            Kernel distance measure to use

        Returns
        -------
        numpy.ndarray
            Matrix of kernel distances between each pairing of inputs

        """
        if isinstance(kern, Gaussian):
            if self.sigma == -99:
                sqd = np.zeros((len(x1), len(x1)))
                for i in range(len(x1)):
                    for j in range(len(x1)):
                        sqd[i, j] = np.linalg.norm(x1[i] - x2[j])
                self.sigma = np.median(sqd)
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j], sigma=self.sigma)
            return k
        else:
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j])
            return k


class Classifier:
    def __init__(self):
        self.b = 0
        self.alphas = []

        self.kern = None
        self.ys = None
        self.xs = None
        self.sigma = -99

        self.support_vectors = []

    def predict(self, x):
        gmx = self.gram_matrix(x, self.xs, self.kern)
        return np.sign(np.sum(np.multiply((np.multiply(self.alphas, self.ys[:, None])), gmx[:, None]), axis=0) + self.b)

    def gram_matrix(self, x1, x2, kern):
        """
        Return matrix comparing x1 to x2 using kern

        Parameters
        ----------
        x1 : numpy.ndarray
            Feature arrays representing data-points / class labels
        x2 : numpy.ndarray
            Feature arrays representing data-points / class labels
        kern : Kernel
            Kernel distance measure to use

        Returns
        -------
        numpy.ndarray
            Matrix of kernel distances between each pairing of inputs

        """
        if isinstance(kern, Gaussian):
            if self.sigma == -99:
                sqd = np.zeros((len(x2), len(x2)))
                for i in range(len(x2)):
                    for j in range(len(x2)):
                        sqd[i, j] = np.linalg.norm(x2[i] - x2[j])
                self.sigma = np.median(sqd)
            k = np.zeros(len(x2))
            for i in range(len(x2)):
                k[i] = kern(x1, x2[i], sigma=self.sigma)
            return k
        else:
            k = np.zeros(len(x2))
            for i in range(len(x2)):
                k[i] = kern(x1, x2[i])
            return k

    def f(self, x):
        gmx = self.gram_matrix(x, self.xs, self.kern)
        return np.sum(np.multiply((np.multiply(self.alphas, self.ys[:, None])), gmx[:, None]), axis=0) + self.b


class SvmUProblem:
    """
    All information about a problem that can be solved by SVMu

    Attributes
    ----------
    C : float
        The regularization parameter to use for this problem
    gamma : float
        Parameter controlling that priority between X and X*
    sigma : float
        Parameter controlling that priority between X and X**
    delta : float
        Parameter controlling the strictness of `classification' in X**
    xkernel: Kernel
        Kernel used in X
    xskernel: Kernel
        Kernel used in X*
    xsskernel: Kernel
        Kernel used in X**
    gaussian_sigma: float
        Standard deviation in distance between datapoints in space where Gaussian kernel is applied
    X : numpy.array
        Non-privileged data-points
    Xstar : numpy.array
        Privileged data-points
    XstarStar : numpy.array
        Privileged data-points
    Y : numpy.array
        Class labels
    num : int
        Number of data-points
    dimensions: int
        Number of features in X
    xi_xj : numpy.ndarray
        K(x,x) Matrix of kernel distances between each data-point in X
    xstari_xstarj : numpy.ndarray
        K*(x*,x*) Matrix of kernel distances between each data-point in X*
    xstarstari_xstarstarj : numpy.ndarray
        K**(x**,x**) Matrix of kernel distances between each data-point in X**
    yi_yj : numpy.array
        Array of Yi * Yj

    """
    def __init__(self, x, xstar, xstarstar, y, c=1.0, gamma=1.0, sigma=1, delta=1.0, xkernel=Linear(),
                 xskernel=Linear(), xsskernel=Linear()):
        self.C = c
        self.gamma = gamma
        self.sigma = sigma
        self.delta = delta
        self.xkernel = xkernel
        self.xskernel = xskernel
        self.xsskernel = xsskernel
        self.gaussian_sigma = -99

        if isinstance(x, np.ndarray):
            self.X = x
        else:
            self.X = np.array(x)
        if isinstance(xstar, np.ndarray):
            self.Xstar = xstar
        else:
            self.Xstar = np.array(xstar)
        if isinstance(xstarstar, np.ndarray):
            self.XstarStar = xstarstar
        else:
            self.XstarStar = np.array(xstarstar)
        if isinstance(y, np.ndarray):
            self.Y = y
            self.Y = np.asarray(self.Y).reshape(-1)
        else:
            self.Y = np.array(y)
            self.Y = np.asarray(self.Y).reshape(-1)

        self.num = len(self.X)
        self.dimensions = len(self.X[0])
        self.xi_xj = self.gram_matrix(self.X, self.X, self.xkernel)
        self.xstari_xstarj = self.gram_matrix(self.Xstar, self.Xstar, self.xskernel)
        self.xstarstari_xstarstarj = self.gram_matrix(self.XstarStar, self.XstarStar, self.xsskernel)
        self.yi_yj = self.gram_matrix(self.Y, self.Y, Linear())

    def gram_matrix(self, x1, x2, kern):
        """
        Return matrix comparing x1 to x2 using kern

        Parameters
        ----------
        x1 : numpy.ndarray
            Feature arrays representing data-points / class labels
        x2 : numpy.ndarray
            Feature arrays representing data-points / class labels
        kern : Kernel
            Kernel distance measure to use

        Returns
        -------
        numpy.ndarray
            Matrix of kernel distances between each pairing of inputs

        """
        if isinstance(kern, Gaussian):
            if self.gaussian_sigma == -99:
                sqd = np.zeros((len(x1), len(x1)))
                for i in range(len(x1)):
                    for j in range(len(x1)):
                        sqd[i, j] = np.linalg.norm(x1[i] - x2[j])
                self.gaussian_sigma = np.median(sqd)
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j], sigma=self.gaussian_sigma)
            return k
        else:
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j])
            return k


class SvmUProblemTuple:
    """
    All information about a problem that can be solved by SVMu

    Attributes
    ----------
    C : float
        The regularization parameter to use for this problem
    gamma : float
        Parameter controlling that priority between X and X*
    sigma : float
        Parameter controlling that priority between X and X**
    delta : float
        Parameter controlling the strictness of `classification' in X**
    xkernel: Kernel
        Kernel used in X
    xskernel: Kernel
        Kernel used in X*
    xsskernel: Kernel
        Kernel used in X**
    gaussian_sigma: float
        Standard deviation in distance between datapoints in space where Gaussian kernel is applied
    X : numpy.array
        Non-privileged data-points
    Xstar : numpy.array
        Privileged data-points
    XstarStar : numpy.array
        Privileged data-points
    Y : numpy.array
        Class labels
    num : int
        Number of data-points
    dimensions: int
        Number of features in X
    xi_xj : numpy.ndarray
        K(x,x) Matrix of kernel distances between each data-point in X
    xstari_xstarj : numpy.ndarray
        K*(x*,x*) Matrix of kernel distances between each data-point in X*
    xstarstari_xstarstarj : numpy.ndarray
        K**(x**,x**) Matrix of kernel distances between each data-point in X**
    yi_yj : numpy.array
        Array of Yi * Yj

    """
    def __init__(self, p):
        self.C = p[5]
        self.gamma = p[7]
        self.sigma = p[8]
        self.delta = p[6]
        self.xkernel = p[9]
        self.xSkernel = p[10]
        self.xSSkernel = p[11]
        self.gaussian_sigma = -99

        if isinstance(p[0], np.ndarray):
            self.X = p[0]
        else:
            self.X = np.array(p[0])
        if isinstance(p[1], np.ndarray):
            self.Xstar = p[1]
        else:
            self.Xstar = np.array(p[1])
        if isinstance(p[1], np.ndarray):
            self.XstarStar = p[1]
        else:
            self.XstarStar = np.array(p[1])
        if isinstance(p[2], np.ndarray):
            self.Y = p[2]
            self.Y = np.asarray(self.Y).reshape(-1)
        else:
            self.Y = np.array(p[2])
            self.Y = np.asarray(self.Y).reshape(-1)

        self.num = len(self.X)
        self.dimensions = len(self.X[0])
        self.xi_xj = self.gram_matrix(self.X, self.X, self.xkernel)
        self.xstari_xstarj = self.gram_matrix(self.Xstar, self.Xstar, self.xSkernel)
        self.xstarstari_xstarstarj = self.gram_matrix(self.XstarStar, self.XstarStar, self.xSSkernel)
        self.yi_yj = self.gram_matrix(self.Y, self.Y, Linear())

    def gram_matrix(self, x1, x2, kern):
        """
        Return matrix comparing x1 to x2 using kern

        Parameters
        ----------
        x1 : numpy.ndarray
            Feature arrays representing data-points / class labels
        x2 : numpy.ndarray
            Feature arrays representing data-points / class labels
        kern : Kernel
            Kernel distance measure to use

        Returns
        -------
        numpy.ndarray
            Matrix of kernel distances between each pairing of inputs

        """
        if isinstance(kern, Gaussian):
            if self.gaussian_sigma == -99:
                sqd = np.zeros((len(x1), len(x1)))
                for i in range(len(x1)):
                    for j in range(len(x1)):
                        sqd[i, j] = np.linalg.norm(x1[i] - x2[j])
                self.gaussian_sigma = np.median(sqd)
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j], sigma=self.gaussian_sigma)
            return k
        else:
            k = np.zeros((len(x1), len(x1)))
            for i in range(len(x1)):
                for j in range(len(x1)):
                    k[i, j] = kern(x1[i], x2[j])
            return k
