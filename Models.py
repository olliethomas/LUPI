from Container_Classes import *
import numpy as np
from cvxopt import matrix, solvers
from sklearn.svm import SVR


class SVM:
    """
    Vanilla Soft-Margin SVM implementation

    Optimization problem solved using CVXOPT.

    """

    @staticmethod
    def get_name():
        """
        Return name of training model.

        Returns
        -------
        String
            The name of the training model

        """
        return "SVM"

    @staticmethod
    def train(x, prob: SvmProblem):
        """
        Train the model with a given set of training data.

        Parameters
        ----------
        x : numpy.array
            Explicitly stated dataset to allow SVM to be used in a variety of cases.
        prob : SvmProblem
            The problem on which to train the model. Contains hyper-param settings and all training data.

        Returns
        -------
        Classifier
            Classifier object trained by this training model

        """

        # Define the inputs to CVXOPT - See Appendix G.1
        P = np.zeros((prob.num, prob.num))
        for i in range(prob.num):
            for j in range(prob.num):
                P[i][j] = prob.Y[i] * prob.Y[j] * prob.xkernel(x[i], x[j])
        q = -np.ones((prob.num, 1))
        G1 = -np.eye(prob.num)
        G2 = np.eye(prob.num)
        G = np.vstack((G1, G2))
        h1 = np.zeros(prob.num).reshape(-1, 1)
        h2 = np.repeat(prob.C, prob.num).reshape(-1, 1)
        h = np.vstack((h1, h2))
        a = prob.Y.reshape(1, -1)
        b = np.zeros(1)

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(a, tc='d')
        b = matrix(b, tc='d')

        # Solve optimization problem using CVXOPT
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        # Get the bias
        bacond1 = (alphas > 1e-8)
        bacond2 = (alphas <= prob.C)
        bcond = np.array([a and b for a, b in zip(bacond1, bacond2)]).flatten()
        yS = prob.Y[bcond]
        xS = x[bcond]
        aS = alphas[bcond]
        sum_total = 0
        for s in range(len(yS)):
            inner_total = 0
            for m in range(len(yS)):
                am = aS[m]
                ym = yS[m]
                xm_xs = prob.xkernel(xS[m], xS[s])
                inner_total += am * ym * xm_xs
            sum_total += yS[s] - inner_total
        bias = sum_total / len(yS) if len(yS) > 0 else [0]

        # Populate Classifier object to be returned
        clf = Classifier()
        clf.b = bias[0]
        clf.alphas = alphas
        clf.xs = x
        clf.ys = prob.Y
        clf.kern = prob.xkernel
        clf.support_vectors = x[bacond1.flatten()]
        return clf


class SVMp:
    """
    SVM+ implementation.

    Optimization problem solved using CVXOPT.

    Attributes
    ----------
    prob : SvmProblem
        The training examples and hyper-params for which we are training the model to
    alphas : numpy.array
        Learned Lagrange multiplier for each training data-point
    deltas : numpy.array
        Learned for each trained data-point. Represents Beta - C

    """
    def __init__(self):
        self.prob = None
        self.alphas = None
        self.deltas = None

    @staticmethod
    def get_name():
        """
        Return name of training model.

        Returns
        -------
        String
            The name of the training model

        """
        return "SVM+"

    def train(self, prob: SvmProblem):
        """
        Train the model with a given set of training data.

        Parameters
        ----------
        prob : SvmProblem
            The problem on which to train the model. Contains hyper-param settings and all training data.

        Returns
        -------
        Classifier
            Classifier object trained by this training model

        """
        # Define variables
        self.prob = prob
        C = self.prob.C
        gamma = self.prob.gamma

        # Define the inputs to CVXOPT - See Appendix G.2
        P1 = (self.prob.xi_xj * self.prob.yi_yj) + gamma * self.prob.xstari_xstarj
        P2 = gamma * self.prob.xstari_xstarj
        P11 = np.hstack((P1, P2))
        P22 = np.hstack((P2, P2))
        P = np.vstack((P11, P22))
        q = np.hstack((np.repeat(-1, self.prob.num), np.zeros(self.prob.num)))
        negative_eye = -np.eye(self.prob.num, dtype='d')
        zeros = np.zeros((self.prob.num, self.prob.num))
        g1 = np.hstack((negative_eye, zeros))
        g2 = np.hstack((zeros, negative_eye))
        G = np.vstack((g1, g2))
        h1 = np.zeros((self.prob.num, 1))
        h2 = np.repeat(C, self.prob.num).reshape(-1, 1)
        h = np.vstack((h1, h2))
        A1 = np.hstack((self.prob.Y, np.zeros(self.prob.num)))
        A2 = np.repeat(-1, 2 * self.prob.num)
        A = np.vstack((A1, A2))
        b = np.zeros(2)
        b = b.reshape(-1, 1)

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')

        # Solve optimization problem using CVXOPT
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas_deltas = np.array(sol['x'])
        self.alphas = np.asarray(alphas_deltas[:self.prob.num])
        self.deltas = alphas_deltas[self.prob.num:]

        # Populate Classifier object to be returned
        clf = Classifier()
        clf.b = self.getB()
        clf.alphas = self.alphas
        clf.xs = self.prob.X
        clf.ys = self.prob.Y
        clf.kern = self.prob.xkernel
        return clf

    @property
    def nPos(self):
        """
        Get the number of positive support vectors

        Returns
        -------
        int
            The number of positive support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 and self.prob.Y[i] == 1:
                running_total += 1
        return running_total if running_total > 0 else 1

    @property
    def nNeg(self):
        """
        Get the number of negative support vectors

        Returns
        -------
        int
            The number of negative support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 and self.prob.Y[i] == -1:
                running_total += 1
        return running_total if running_total > 0 else 1

    def getB(self):
        """
        Calculates the bias

        Returns
        -------
        float
            The bias for the classifier

        """
        return ((self.bPlusbStar() / self.nPos) + (self.bMinusbStar / self.nNeg)) / 2

    def bPlusbStar(self):
        """
        Calculates the value of b + b* - See equations 4.5 and 4.6

        Returns
        -------
        float
            The average summed value of the biases in X and X*

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 and self.prob.Y[i] == 1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.prob.Y[j] * self.prob.xkernel(self.prob.X[i], self.prob.X[j])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.prob.X[i], self.prob.X[j])
                abcxx *= (1 / self.prob.gamma)
                running_total += 1 - abcxx - ayxx
        return running_total

    @property
    def bMinusbStar(self):
        """
        Calculates the value of b - b* - See equations 4.5 and 4.6

        Returns
        -------
        float
            The average difference of the value of the biases in X and X*

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 and self.prob.Y[i] == -1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.prob.Y[j] * self.prob.xkernel(self.prob.X[i], self.prob.X[j])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.prob.X[i], self.prob.X[j])
                abcxx *= (1 / self.prob.gamma)
                running_total += -1 + abcxx - ayxx
        return running_total


class SVMdpSimp:
    """
    SVM delta+: Simplified Approach implementation.

    Optimization problem solved using CVXOPT.

    """

    @staticmethod
    def get_name():
        """
        Return name of training model.

        Returns
        -------
        String
            The name of the training model

        """
        return "SVMd+ - simplified approach"

    @staticmethod
    def train(prob: SvmProblem):
        """
        Train the model with a given set of training data.

        Parameters
        ----------
        prob : SvmProblem
            The problem on which to train the model. Contains hyper-param settings and all training data.

        Returns
        -------
        Classifier
            Classifier object trained by this training model

        """
        # Define variables
        x = prob.X
        y = prob.Y
        C = prob.C
        C2 = prob.gamma

        # Swap params, so SVM solves X* with correct params
        xk = prob.xkernel
        xsk = prob.xskernel

        prob.C = C2
        prob.xkernel = xsk

        svm = SVM()
        xstar_clf = svm.train(prob.Xstar, prob)

        # Get distance to decision boundary
        xi_star = np.zeros(prob.num)
        for i in range(prob.num):
            output = (1 - prob.Y[i] * (xstar_clf.f(prob.Xstar[i])))
            xi_star[i] = max(0, output)

        # Replace swapped out params so modified SVM solves X with correct params
        prob.C = C
        prob.xkernel = xk

        # Define the inputs to CVXOPT - See Appendix G.4
        P = prob.yi_yj * prob.xi_xj
        q = -np.ones((prob.num, 1))
        G1 = -np.eye(prob.num)
        G2 = np.eye(prob.num)
        G3 = xi_star.reshape(1, -1)
        G = np.vstack((G1, G2))
        G = np.vstack((G, G3))
        h1 = np.zeros(prob.num).reshape(-1, 1)
        h2 = np.repeat((1 + prob.delta) * C, prob.num).reshape(-1, 1)
        h3 = sum(xi_star) * C
        h = np.vstack((h1, h2))
        h = np.vstack((h, h3))
        A = y.reshape(1, -1)
        b = np.zeros(1)

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')

        # Solve optimization problem using CVXOPT
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        # Get the bias
        bacond1 = (alphas > 1e-8)
        bacond2 = (alphas <= (1 + prob.delta) * C)
        bcond = np.array([a and b for a, b in zip(bacond1, bacond2)]).flatten()

        yS = y[bcond]
        xS = x[bcond]
        aS = alphas[bcond]

        sum_total = 0
        for s in range(len(yS)):
            inner_total = 0
            for m in range(len(yS)):
                am = aS[m]
                ym = yS[m]
                xm_xs = prob.xkernel(xS[m], xS[s])
                inner_total += am * ym * xm_xs
            sum_total += yS[s] - inner_total

        bias = sum_total / len(yS)

        # Populate Classifier object to be returned
        clf = Classifier()
        clf.b = bias
        clf.alphas = alphas
        clf.xs = x
        clf.ys = y
        clf.kern = prob.xkernel
        return clf


class SVMdp:
    """
    SVM delta+ implementation.

    Optimization problem solved using CVXOPT.

    Attributes
    ----------
    prob : SvmProblem
        The training examples and hyper-params for which we are training the model to
    alphas : numpy.array
        Learned Lagrange multiplier for each training data-point
    deltas : numpy.array
        Learned for each trained data-point. Represents Beta - C

    """
    def __init__(self):
        self.prob = None
        self.alphas = None
        self.deltas = None

    @staticmethod
    def get_name():
        """
        Return name of training model.

        Returns
        -------
        String
            The name of the training model

        """
        return "SVMd+"

    def train(self, prob: SvmProblem):
        """
        Train the model with a given set of training data.

        Parameters
        ----------
        prob : SvmProblem
            The problem on which to train the model. Contains hyper-param settings and all training data.

        Returns
        -------
        Classifier
            Classifier object trained by this training model

        """
        # Define variables
        self.prob = prob
        x = prob.X
        y = prob.Y

        # Define the inputs to CVXOPT - See Appendix G.3
        P11 = (prob.xi_xj * prob.yi_yj) + prob.gamma * (prob.xstari_xstarj * prob.yi_yj)
        P12 = prob.gamma * (prob.xstari_xstarj * prob.yi_yj)
        P1 = np.hstack((P11, P12))
        P2 = np.hstack((P12, P12))
        P = np.vstack((P1, P2))
        q = np.hstack((np.repeat(-1, prob.num), np.zeros(prob.num)))
        positive_eye = np.eye(prob.num, dtype='d')
        negative_eye = -np.eye(prob.num, dtype='d')
        zeros = np.zeros((prob.num, prob.num))
        g1 = np.hstack((negative_eye, zeros))
        g2 = np.hstack((positive_eye, positive_eye))
        g3 = np.hstack((zeros, negative_eye))
        G = np.vstack((g1, g2))
        G = np.vstack((G, g3))
        h1 = np.zeros((prob.num, 1))
        h2 = np.repeat((prob.delta * prob.C), prob.num).reshape(-1, 1)
        h3 = np.repeat(prob.C, prob.num).reshape(-1, 1)
        h = np.vstack((h1, h2))
        h = np.vstack((h, h3))
        A1 = np.hstack((prob.Y, np.zeros(prob.num)))
        A2 = np.hstack((-prob.Y, -prob.Y))
        A = np.vstack((A1, A2))
        b = np.zeros(2)
        b = b.reshape(-1, 1)

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')

        # Solve optimization problem using CVXOPT
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas_and_deltas = np.array(sol['x'])
        self.alphas = np.asarray(alphas_and_deltas[:prob.num])
        self.deltas = alphas_and_deltas[prob.num:]

        # Populate Classifier object to be returned
        clf = Classifier()
        clf.b = self.get_b
        clf.alphas = self.alphas
        clf.xs = x
        clf.ys = y
        clf.kern = prob.xkernel
        return clf

    @property
    def s_pos(self):
        """
        Get the sum of [1 - <w,X>] for positive support vectors

        Returns
        -------
        int
            The sum of [1 - <w,X>] for positive support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 > self.prob.C - self.deltas[i] and self.prob.Y[i] == 1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.prob.Y[j] * self.prob.xkernel(self.prob.X[j], self.prob.X[i])
                running_total += 1 - ayxx
        return running_total

    @property
    def s_neg(self):
        """
        Get the sum of [-1 - <w,X>] for negative support vectors

        Returns
        -------
        int
            The sum of [-1 - <w,X>] for negative support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 > self.prob.C - self.deltas[i] and self.prob.Y[i] == -1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.prob.Y[j] * self.prob.xkernel(self.prob.X[j], self.prob.X[i])
                running_total += -1 - ayxx
        return running_total

    @property
    def n_pos(self):
        """
        Get the number of positive support vectors

        Returns
        -------
        int
            The number of positive support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 > self.prob.C - self.deltas[i] and self.prob.Y[i] == 1:
                running_total += 1
        return running_total if running_total > 0 else 1

    @property
    def n_neg(self):
        """
        Get the number of negative support vectors

        Returns
        -------
        int
            The number of negative support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 > self.prob.C - self.deltas[i] and self.prob.Y[i] == -1:
                running_total += 1
        return running_total if running_total > 0 else 1

    @property
    def get_b(self):
        """
        Calculates the bias

        Returns
        -------
        float
            The bias for the classifier

        """
        return ((self.s_pos / self.n_pos) + (self.s_neg / self.n_neg)) / 2


class SVMu:
    """
    SVM+ implementation.

    Optimization problem solved using CVXOPT.

    Attributes
    ----------
    prob : SvmProblem
        The training examples and hyper-params for which we are training the model to
    alphas : numpy.array
        Learned Lagrange multiplier for each training data-point
    etas : nupmy.array
        Learned Lagrange multiplier for each training data-point
    deltas : numpy.array
        Learned for each trained data-point. Represents Lagrange multiplier Beta - C
    epsilons : numpy.array
        Learned for each trained data-point. Represents Lagrange multiplier Mu - C

    """
    def __init__(self):
        self.prob = None
        self.alphas = None
        self.etas = None
        self.deltas = None
        self.epsilons = None

    @staticmethod
    def get_name():
        """
        Return name of training model.

        Returns
        -------
        String
            The name of the training model

        """
        return "SVM Idea"

    def train(self, prob: SvmUProblem):
        """
        Train the model with a given set of training data.

        Parameters
        ----------
        prob : SvmProblem
            The problem on which to train the model. Contains hyper-param settings and all training data.

        Returns
        -------
        Classifier
            Classifier object trained by this training model

        """
        # Define variables
        self.prob = prob
        x = self.prob.X
        y = self.prob.Y

        # Define the inputs to CVXOPT - See Appendix G.6
        P_a = prob.yi_yj * prob.xi_xj
        P_b = prob.gamma * prob.xstari_xstarj
        P_c = prob.sigma * prob.yi_yj * prob.xstarstari_xstarstarj

        P_row_0 = np.hstack((P_a + P_b, P_a))
        P_row_0 = np.hstack((P_row_0, P_b))
        P_row_0 = np.hstack((P_row_0, np.zeros((prob.num, prob.num))))
        P_row_1 = np.hstack((P_a, P_a + P_c))
        P_row_1 = np.hstack((P_row_1, np.zeros((prob.num, prob.num))))
        P_row_1 = np.hstack((P_row_1, P_c))
        P_row_2 = np.hstack((P_b, np.zeros((prob.num, prob.num))))
        P_row_2 = np.hstack((P_row_2, P_b))
        P_row_2 = np.hstack((P_row_2, np.zeros((prob.num, prob.num))))
        P_row_3 = np.hstack((np.zeros((prob.num, prob.num)), P_c))
        P_row_3 = np.hstack((P_row_3, np.zeros((prob.num, prob.num))))
        P_row_3 = np.hstack((P_row_3, P_c))

        P = np.vstack((P_row_0, P_row_1))
        P = np.vstack((P, P_row_2))
        P = np.vstack((P, P_row_3))

        q = np.hstack((np.repeat(-1, prob.num), np.repeat(-1, prob.num)))
        q = np.hstack((q, np.zeros(prob.num)))
        q = np.hstack((q, np.zeros(prob.num)))

        positive_eye = np.eye(prob.num, dtype='d')
        negative_eye = -np.eye(prob.num, dtype='d')
        zeros = np.zeros((prob.num, prob.num))

        # g1 = -a <= 0
        g1 = np.hstack((negative_eye, zeros))
        g1 = np.hstack((g1, zeros))
        g1 = np.hstack((g1, zeros))

        # g2 = -n <= 0
        g2 = np.hstack((zeros, negative_eye))
        g2 = np.hstack((g2, zeros))
        g2 = np.hstack((g2, zeros))

        # g3 = -d <= C
        g3 = np.hstack((zeros, zeros))
        g3 = np.hstack((g3, negative_eye))
        g3 = np.hstack((g3, zeros))

        # g4 = n + e <= Del C
        g4 = np.hstack((zeros, positive_eye))
        g4 = np.hstack((g4, zeros))
        g4 = np.hstack((g4, positive_eye))

        # g5 = -e <= C
        g5 = np.hstack((zeros, zeros))
        g5 = np.hstack((g5, zeros))
        g5 = np.hstack((g5, negative_eye))

        G = np.vstack((g1, g2))
        G = np.vstack((G, g3))
        G = np.vstack((G, g4))
        G = np.vstack((G, g5))

        h1 = np.zeros((prob.num, 1))
        h2 = np.zeros((prob.num, 1))
        h3 = np.repeat(prob.C, prob.num).reshape(-1, 1)
        h4 = np.repeat((prob.delta * prob.C), prob.num).reshape(-1, 1)
        h5 = np.repeat(prob.C, prob.num).reshape(-1, 1)
        h = np.vstack((h1, h2))
        h = np.vstack((h, h3))
        h = np.vstack((h, h4))
        h = np.vstack((h, h5))

        A1 = np.hstack((prob.Y, -prob.Y))
        A1 = np.hstack((A1, np.zeros(2 * prob.num)))
        A2 = np.hstack((-np.ones(prob.num), np.zeros(prob.num)))
        A2 = np.hstack((A2, -np.ones(prob.num)))
        A2 = np.hstack((A2, np.zeros(prob.num)))
        A3 = np.hstack((np.zeros(prob.num), -prob.Y))
        A3 = np.hstack((A3, np.zeros(prob.num)))
        A3 = np.hstack((A3, -prob.Y))

        A = np.vstack((A1, A2))
        A = np.vstack((A, A3))

        b = np.zeros(3)
        b = b.reshape(-1, 1)

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')

        # Solve optimization problem using CVXOPT
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas_etas_deltas_epsilons = np.array(sol['x'])
        self.alphas = alphas_etas_deltas_epsilons[:prob.num]
        self.etas = alphas_etas_deltas_epsilons[prob.num:(2 * prob.num)]
        self.deltas = alphas_etas_deltas_epsilons[(2 * prob.num):(3 * prob.num)]
        self.epsilons = alphas_etas_deltas_epsilons[-prob.num:]

        # Populate Classifier object to be returned
        clf = Classifier()
        clf.b = self.get_b
        clf.alphas = np.asarray(self.alphas + self.etas)
        clf.xs = x
        clf.ys = y
        clf.kern = prob.xkernel
        return clf

    @property
    def n_pos(self):
        """
        Get the number of positive support vectors

        Returns
        -------
        int
            The number of positive support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 > self.prob.C - self.deltas[i] and self.prob.Y[i] == 1:
                running_total += 1
        return running_total if running_total > 0 else 1

    @property
    def n_neg(self):
        """
        Get the number of negative support vectors

        Returns
        -------
        int
            The number of negative support vectors

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 > self.prob.C - self.deltas[i] and self.prob.Y[i] == -1:
                running_total += 1
        return running_total if running_total > 0 else 1

    @property
    def get_b(self):
        """
        Calculates the bias - using method from SVM+

        Returns
        -------
        float
            The bias for the classifier

        """
        return ((self.b_plus_bstar / self.n_pos) + (self.b_minus_bstar / self.n_neg)) / 2

    @property
    def b_plus_bstar(self):
        """
        Calculates the value of b + b* - See equations 4.5 and 4.6

        Returns
        -------
        float
            The average summed value of the biases in X and X*

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 and self.prob.Y[i] == 1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += (self.alphas[j] + self.etas[j]) * self.prob.Y[j] * self.prob.xkernel(self.prob.X[j],
                                                                                                 self.prob.X[i])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.prob.X[j], self.prob.X[i])
                abcxx *= (1 / self.prob.gamma)
                running_total += 1 - abcxx - ayxx
        return running_total

    @property
    def b_minus_bstar(self):
        """
        Calculates the value of b - b* - See equations 4.5 and 4.6

        Returns
        -------
        float
            The average difference of the value of the biases in X and X*

        """
        running_total = 0
        for i in range(self.prob.num):
            if self.alphas[i] > 1e-5 and self.prob.Y[i] == -1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += (self.alphas[j] + self.etas[j]) * self.prob.Y[j] * self.prob.xkernel(self.prob.X[j],
                                                                                                 self.prob.X[i])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.prob.X[j], self.prob.X[i])
                abcxx *= (1 / self.prob.gamma)
                running_total += -1 + abcxx - ayxx
        return running_total


class KT:
    """
    Naive implementation of Knowledge Transfer.

    Naive due to lack of optimization and hyper-param tuning - particularly around the regression

    Attributes
    ----------
    prob : SvmProblem
        The training examples and hyper-params for which we are training the model to
    models : [sklearn.svm.SVR]
        List of regression models
    clf : SVMdp
        Learned classifier

    """
    def __init__(self):
        self.prob = None
        self.models = None
        self.clf = None

    def train(self, prob: SvmProblem):
        """
        Train the model with a given set of training data.

        Parameters
        ----------
        prob : SvmProblem
            The problem on which to train the model. Contains hyper-param settings and all training data.

        """
        # Define variables
        self.prob = prob
        c = prob.C
        c2 = prob.gamma
        xkern = prob.xkernel
        xsk = prob.xskernel

        # Swap params, so SVM solves X* with correct params
        prob.C = c2
        prob.xkernel = xsk

        # Find SVM solution in X*
        svm = SVM()
        priv_clf = svm.train(prob.Xstar, prob)

        # Replace swapped out params so modified SVM solves X with correct params
        prob.C = c
        prob.xkernel = xkern

        # Get the 'frames of knowledge' - Get the kernel distance from each
        # privileged training data-point to the privileged support vectors
        frames = np.zeros((prob.num, len(priv_clf.support_vectors)))
        for i in range(prob.num):
            for j in range(len(priv_clf.support_vectors)):
                frames[i][j] = prob.xkernel((priv_clf.support_vectors[j]), prob.Xstar[i])

        # Form pairs so that each training point is matched against each 'frame of knowledge'
        training_pairs = np.zeros((prob.num, len(priv_clf.support_vectors)), dtype=object)
        for i in range(prob.num):
            for j in range(len(priv_clf.support_vectors)):
                training_pairs[i][j] = [prob.X[i], frames[i][j]]
        training_pairs = np.array(training_pairs)

        regr_pairs = np.zeros((len(priv_clf.support_vectors), prob.num), dtype=object)
        for i in range(prob.num):
            for j in range(len(priv_clf.support_vectors)):
                regr_pairs[j][i] = training_pairs[i][j]

        # Learn a regression based on above pairs
        self.models = []
        for dataSet in regr_pairs:
            regr = SVR(kernel='rbf')
            xs = []
            ys = []
            for i in range(prob.num):
                xs.append(dataSet[i][0].flatten())
                ys.append(dataSet[i][1])
            xs = np.array(xs)
            ys = np.array(ys)
            self.models.append(regr.fit(xs, ys))

        # Transform data from X using learned regression
        new_xs = []
        new_ys = []
        for i in range(prob.num):
            new_xs.append(self.transform(prob.X[i].reshape(1, -1)).flatten())
            new_ys.append(priv_clf.predict(prob.Xstar[i]))
        new_x = np.asarray(new_xs)
        new_y = np.array(new_ys)

        # Form a new problem and learn an SVMd+ solution for it
        new_prob = SvmProblem(new_x, prob.Xstar, new_y)
        new_svm = SVMdp()
        self.clf = new_svm.train(new_prob)

    def transform(self, x):
        """
        Transform a data-point using regression learned mapping X to 'frame of knowledge'

        Parameters
        ----------
        x : numpy.array
            Data-point to be transformed

        Returns
        -------
        numpy.array
            Data-point after transformation

        """
        to_return = []
        for i in range(len(self.models)):
            to_return.append(self.models[i].predict(x))
        return np.array(to_return)

    def predict(self, x):
        """
        Transform a data-point to learned space, then classify.

        Parameters
        ----------
        x : numpy.array
            The data-point to predict the classification of

        Returns
        -------
        int
            1 for +ve class, else -1

        """
        new_x = np.array(self.transform(x.reshape(1, -1)).flatten())
        return self.clf.predict(new_x.T)
