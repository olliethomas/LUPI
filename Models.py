from Container_Classes import *
import numpy as np
from cvxopt import matrix, solvers
from numpy.linalg import matrix_rank as mr
import logging
from sklearn.svm import SVR

class SVM():
    def get_name(self):
        return "SVM"
    def train(self, x, prob : svm_problem):
        logging.info("Entered Train")
        x = x
        y = prob.Y
        C = prob.C

        NUM = x.shape[0]
        DIM = x.shape[1]

        K = y[:, None] * x # Yeah, this is a bit different so that it can work on x and x*
        K = np.dot(K, K.T)
        P = matrix(K, tc='d')
        q = matrix(-np.ones((NUM, 1)), tc='d')
        G1 = -np.eye(NUM)
        G2 = np.eye(NUM)
        G = np.vstack((G1, G2))
        G = matrix(G, tc='d')
        h1 = np.zeros(NUM).reshape(-1,1)
        h2 = np.repeat(C, NUM).reshape(-1,1)
        h = np.vstack((h1, h2))
        h = matrix(h, tc='d')
        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(np.zeros(1), tc='d')
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        w = np.sum(alphas * y[:, None] * x, axis = 0)
        bacond1 = (alphas > 1e-8)
        bacond2 = (alphas <= (C))
        bcond = np.array([a and b for a, b in zip(bacond1, bacond2)]).flatten()
        yS = y[bcond]
        xS = x[bcond]
        aS = alphas[bcond]
        sumTotal = 0
        for s in range(len(yS)):
            innerTotal = 0
            for m in range(len(yS)):
                am = aS[m]
                ym = yS[m]
                xm_xs = prob.xkernel(xS[m], xS[s])
                innerTotal += am*ym*xm_xs
            sumTotal += yS[s] - innerTotal
        bias = sumTotal/len(yS) if len(yS) > 0 else [0]
        clf = classifier()
        clf.w = w
        clf.b = bias[0]
        clf.alphas = alphas
        clf.xs = x
        clf.ys = y
        clf.kern = prob.xkernel
        clf.support_vectors = x[bacond1.flatten()]
        return clf

class SVMp():
    def get_name(self):
        return "SVM+"
    def train(self, prob : svm_problem):
        self.prob = prob
        self.C = self.prob.C

        self.L = self.prob.num

        self.x = self.prob.X
        self.xStar = self.prob.Xstar
        self.y = self.prob.Y

        self.gamma = self.prob.gamma

        P1 = (self.prob.xi_xj * self.prob.yi_yj) + self.gamma*(self.prob.xstari_xstarj)
        P2 = self.gamma*(self.prob.xstari_xstarj)
        P11 = np.hstack((P1, P2))
        P22 = np.hstack((P2, P2))
        P = np.vstack((P11, P22))

        q = np.hstack((np.repeat(-1, self.L),np.zeros(self.L)))

        positiveEye = np.eye(self.L, dtype='d')
        negativeEye = -np.eye(self.L, dtype='d')
        zeros = np.zeros((self.L, self.L))
        g1 = np.hstack((negativeEye, zeros))
        g2 = np.hstack((zeros, negativeEye))

        G = np.vstack((g1,g2))

        h1 = np.zeros(((self.L),1))
        h2 = np.repeat(self.C, (self.L)).reshape(-1,1)
        h = np.vstack((h1, h2))

        A1 = np.repeat(-1, 2*self.L)
        A2 = np.hstack((self.y, np.zeros(self.L)))
        A = np.vstack((A1, A2))

        b = np.zeros(2)
        b = b.reshape(-1,1)

        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A, tc='d')
        b = matrix(b, tc='d')

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphasAndDeltas = np.array(sol['x'])
        self.alphas = np.asarray(alphasAndDeltas[:self.L])
        self.deltas = alphasAndDeltas[self.L:]

        self.w = np.sum(self.alphas * self.y[:, None] * self.x, axis = 0)
        self.wStar = (1/self.gamma)*np.sum((self.alphas + self.deltas) * self.prob.Xstar, axis = 0)

        bacond = (self.alphas > 1e-5)
        bdcond = (self.deltas + self.C > 1e-5)

        bcond = np.array([a and b for a, b in zip(bacond, bdcond)]).flatten()

        clf = classifier()
        clf.w = self.w
        clf.b = self.getB()
        clf.alphas = self.alphas
        clf.xs = self.x
        clf.ys = self.y
        clf.kern = self.prob.xkernel
        clf.support_vectors = self.x[(self.alphas > 1e-5).flatten()]
        clf1 = classifier()
        clf1.w = self.wStar
        clf1.b = self.getBstar()
        clf1.alphas = self.alphas
        clf1.support_vectors = self.prob.Xstar[bcond]
        return clf #, clf1 # Sort of useful for visualising what's going on in X*

    def nPos(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.y[i] == 1:
                runningTotal += 1
        return runningTotal if runningTotal > 0 else 1

    def nNeg(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.y[i] == -1:
                runningTotal += 1
        return runningTotal if runningTotal > 0 else 1

    def getB(self):
        return ((self.bPlusbStar()/self.nPos())+(self.bMinusbStar()/self.nNeg()))/2

    def getBstar(self):
        return ((self.bPlusbStar()/self.nPos())-(self.bMinusbStar()/self.nNeg()))/2

    def bPlusbStar(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.y[i] == 1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.y[j] * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = (1/self.prob.gamma)*abcxx
                runningTotal += 1 - abcxx - ayxx
        return runningTotal


    def bMinusbStar(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.y[i] == -1:
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.y[j] * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = (1/self.prob.gamma)*abcxx
                runningTotal += -1 + abcxx - ayxx
        return runningTotal

class SVMdp_simp():
    def get_name(self):
        return "SVMd+ - simplified approach"
    def train(self, prob : svm_problem):
        x = prob.X
        xStar = prob.Xstar
        y = prob.Y
        C = prob.C

        NUM = x.shape[0]
        DIM = x.shape[1]

        svm = SVM()
        xStar_clf = svm.train(xStar, prob)

        xi_star_amended = np.zeros(prob.num)
        for i in range(prob.num):
            output = (1- prob.Y[i]*(xStar_clf.f(prob.Xstar[i])))
            xi_star_amended[i] = max(0, output)

        Ky = prob.yi_yj
        Kx = prob.xi_xj
        K = Ky*Kx
        P = matrix(K, tc='d')
        q = matrix(-np.ones((NUM, 1)), tc='d')
        G1 = -np.eye(NUM)
        G2 = np.eye(NUM)
        G3 = xi_star_amended.reshape(1,-1)
        G = np.vstack((G1, G2))
        G = np.vstack((G, G3))
        G = matrix(G, tc='d')
        h1 = np.zeros(NUM).reshape(-1,1)
        h2 = np.repeat((1+prob.delta)*C, NUM).reshape(-1,1)
        h3 = sum(xi_star_amended)*C
        h = np.vstack((h1, h2))
        h = np.vstack((h, h3))
        h = matrix(h, tc='d')
        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(np.zeros(1), tc='d')
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        w = np.sum(alphas * y[:, None] * x, axis = 0)

        bacond1 = (alphas > 1e-8)
        bacond2 = (alphas <= (1+prob.delta)*C)
        bcond = np.array([a and b for a, b in zip(bacond1, bacond2)]).flatten()

        yS = y[bcond]
        xS = x[bcond]
        aS = alphas[bcond]

        sumTotal = 0
        for s in range(len(yS)):
            innerTotal = 0
            for m in range(len(yS)):
                am = aS[m]
                ym = yS[m]
                xm_xs = prob.xkernel(xS[m], xS[s])
                innerTotal += am*ym*xm_xs
            sumTotal += yS[s] - innerTotal

        bias = sumTotal/len(yS)

        clf = classifier()
        clf.w = w
        clf.b = bias
        clf.alphas = alphas
        clf.xs = x
        clf.ys = y
        clf.kern = prob.xkernel
        clf.support_vectors = prob.X[bacond1.flatten()]
        return clf

class SVMdp():
    def get_name(self):
        return "SVMd+"
    def train(self, prob : svm_problem):
        self.prob = prob
        #self.kernel = self.prob.kernel
        self.C = self.prob.C

        self.L = self.prob.num
        self.M = self.prob.dimensions

        self.x = self.prob.X
        self.y = self.prob.Y

        self.gamma = self.prob.gamma
        self.delta = self.prob.delta

        C = prob.C

        L = prob.num
        M = prob.dimensions

        x = prob.X
        y = prob.Y

        H11 = (prob.xi_xj * prob.yi_yj) + self.gamma*(prob.xstari_xstarj * prob.yi_yj)
        H12 = self.gamma*(prob.xstari_xstarj * prob.yi_yj)
        H1 = np.hstack((H11, H12))
        H2 = np.hstack((H12, H12))
        H = np.vstack((H1, H2))

        f = np.hstack((np.repeat(-1, L),np.zeros(L)))

        positiveEye = np.eye(L, dtype='d')
        negativeEye = -np.eye(L, dtype='d')
        zeros = np.zeros((L, L))
        g1 = np.hstack((zeros, negativeEye))
        g2 = np.hstack((negativeEye, zeros))
        g3 = np.hstack((positiveEye, positiveEye))

        G = np.vstack((g1,g2))
        G = np.vstack((G,g3))

        h1 = np.repeat(C, (L)).reshape(-1,1)
        h2 = np.zeros(((L),1))
        h2 = np.vstack((h1, h2))
        h3 = np.repeat((self.delta*C), L).reshape(-1,1)
        h = np.vstack((h2, h3))

        Aeq1 = np.hstack((prob.Y, np.zeros(L)))
        Aeq2 = np.hstack((-prob.Y, -prob.Y))
        Aeq = np.vstack((Aeq1, Aeq2))

        beq = np.zeros(2)
        beq = beq.reshape(-1,1)

        P = matrix(H, tc='d')
        q = matrix(f, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(Aeq, tc='d')
        b = matrix(beq, tc='d')

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphasAndDeltas = np.array(sol['x'])
        self.alphas = np.asarray(alphasAndDeltas[:L])
        self.deltas = alphasAndDeltas[L:]

        self.w = np.sum(self.alphas * self.y[:, None] * self.x, axis = 0)
        self.wStar = (1/self.gamma)*np.sum((self.alphas + self.deltas) * self.y[:, None] * self.prob.Xstar, axis = 0)

        bacond = (self.alphas > 1e-5)
        bdcond = (self.deltas + self.C > 1e-5)

        bcond = np.array([a and b for a, b in zip(bacond, bdcond)]).flatten()

        clf = classifier()
        clf.w = self.w
        self.b = self.getB()
        clf.b = self.b
        clf.alphas = self.alphas
        clf.xs = x
        clf.ys = y
        clf.kern = prob.xkernel
        clf.support_vectors = self.x[bacond.flatten()]

        priv_clf = classifier()
        priv_clf.w = self.wStar
        priv_clf.b = self.getBstar()
        priv_clf.support_vectors = self.prob.Xstar[np.array(bacond).flatten()]
        return clf #, priv_clf # Uncomment to get "priv classifier" - useful for understanding in 2d world, else useless

    def sPos(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == 1:
                #runningTotal += 1-np.dot(self.w, self.prob.X[i])
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.y[j] * self.prob.xkernel(self.x[j], self.x[i])
                runningTotal += 1 - ayxx
        return runningTotal

    def sNeg(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == -1:
                #runningTotal += -1 - np.dot(self.w, self.prob.X[i])
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.y[j] * self.prob.xkernel(self.x[j], self.x[i])
                runningTotal += -1 - ayxx
        return runningTotal

    def nPos(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == 1:
                runningTotal += 1
        return runningTotal if runningTotal > 0 else 1

    def nNeg(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == -1:
                runningTotal += 1
        return runningTotal if runningTotal > 0 else 1

    def getB(self):
        return ((self.sPos()/self.nPos())+(self.sNeg()/self.nNeg()))/2

    def q(self):
        runningTotal = 0
        for i in range(self.L):
            if self.deltas[i] + self.C > 1e-5:
                runningTotal += np.dot(self.w, self.prob.X[i])/2 - np.dot(self.wStar, self.prob.Xstar[i])
        return runningTotal

    def getBstar(self):
        return self.q() / self.L

class SVMu():
    def get_name(self):
        return "SVM Idea"
    def train(self, prob : svm_problem):
        self.prob = prob
        self.C = self.prob.C

        self.L = self.prob.num
        self.M = self.prob.dimensions

        self.x = self.prob.X
        self.xS = self.prob.Xstar
        self.xSS = self.prob.XstarStar
        self.y = self.prob.Y

        self.gamma = self.prob.gamma
        self.sigma = self.prob.sigma
        self.delta = self.prob.delta

        C = self.C

        L = self.L
        M = self.M

        x = self.x
        y = self.y


        ha = (prob.yi_yj*prob.xi_xj)
        hb = self.gamma*prob.xstari_xstarj
        hc = self.sigma*prob.xstarstari_xstarstarj

        h00 = ha+hb
        h01 = ha
        h02 = hb
        h03 = np.zeros((L,L))
        h10 = ha
        h11 = ha+hc
        h12 = np.zeros((L,L))
        h13 = hc
        h20 = hb
        h21 = np.zeros((L,L))
        h22 = hb
        h23 = np.zeros((L,L))
        h30 = np.zeros((L,L))
        h31 = hc
        h32 = np.zeros((L,L))
        h33 = hc

        h1strow = np.hstack((h00, h01))
        h1strow = np.hstack((h1strow, h02))
        h1strow = np.hstack((h1strow, h03))
        h2ndrow = np.hstack((h10, h11))
        h2ndrow = np.hstack((h2ndrow, h12))
        h2ndrow = np.hstack((h2ndrow, h13))
        h3rdrow = np.hstack((h20, h21))
        h3rdrow = np.hstack((h3rdrow, h22))
        h3rdrow = np.hstack((h3rdrow, h23))
        h4throw = np.hstack((h30, h31))
        h4throw = np.hstack((h4throw, h32))
        h4throw = np.hstack((h4throw, h33))

        H = np.vstack((h1strow, h2ndrow))
        H = np.vstack((H, h3rdrow))
        H = np.vstack((H, h4throw))

        f = np.hstack((np.repeat(-1, L), np.repeat(-1, L)))
        f = np.hstack((f,np.zeros(L)))
        f = np.hstack((f, np.zeros(L)))


        positiveEye = np.eye(L, dtype='d')
        negativeEye = -np.eye(L, dtype='d')
        zeros = np.zeros((L, L))

        # g1 = -a <= 0
        g1 = np.hstack((negativeEye, zeros))
        g1 = np.hstack((g1, zeros))
        g1 = np.hstack((g1, zeros))

        # g2 = -d <= C
        g2 = np.hstack((zeros, zeros))
        g2 = np.hstack((g2, negativeEye))
        g2 = np.hstack((g2, zeros))

        # g3 = -n <= 0
        g3 = np.hstack((zeros, negativeEye))
        g3 = np.hstack((g3, zeros))
        g3 = np.hstack((g3, zeros))

        # g4 = n + e <= Del C
        g4 = np.hstack((zeros, positiveEye))
        g4 = np.hstack((g4, zeros))
        g4 = np.hstack((g4, positiveEye))

        # g5 = -e <= c
        g5 = np.hstack((zeros, zeros))
        g5 = np.hstack((g5, zeros))
        g5 = np.hstack((g5, negativeEye))

        G = np.vstack((g1,g2))
        G = np.vstack((G,g3))
        G = np.vstack((G, g4))
        G = np.vstack((G, g5))

        h1 = np.zeros(((L),1))
        h2 = np.repeat(C, (L)).reshape(-1,1)
        h3 = np.zeros(((L),1))
        h4 = np.repeat((self.delta*C), L).reshape(-1,1)
        h5 = np.repeat(C, (L)).reshape(-1,1)
        h = np.vstack((h1, h2))
        h = np.vstack((h, h3))
        h = np.vstack((h, h4))
        h = np.vstack((h, h5))

        Aeq1 = np.hstack((prob.Y, -prob.Y))
        Aeq1 = np.hstack((Aeq1, np.zeros(2*L)))
        Aeq2 = np.hstack((-np.ones(L), np.zeros(L)))
        Aeq2 = np.hstack((Aeq2, -np.ones(L)))
        Aeq2 = np.hstack((Aeq2, np.zeros(L)))
        Aeq3 = np.hstack((np.zeros(L), -prob.Y))
        Aeq3 = np.hstack((Aeq3, np.zeros(L)))
        Aeq3 = np.hstack((Aeq3, -prob.Y))

        Aeq = np.vstack((Aeq1, Aeq2))
        Aeq = np.vstack((Aeq, Aeq3))


        beq = np.zeros(3)
        beq = beq.reshape(-1,1)

        P = matrix(H, tc='d')
        q = matrix(f, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(Aeq, tc='d')
        b = matrix(beq, tc='d')

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphasEtasDeltasEpsilons = np.array(sol['x'])
        self.alphas = alphasEtasDeltasEpsilons[:L]
        self.etas = alphasEtasDeltasEpsilons[L:(2*L)]
        self.deltas = alphasEtasDeltasEpsilons[(2*L):(3*L)]
        self.epsilons = alphasEtasDeltasEpsilons[-L:]

        self.w = np.sum((self.alphas+self.etas) * self.y[:, None] * self.x, axis = 0)
        self.wStar = (1/self.gamma)*np.sum((self.alphas + self.deltas) * self.y[:, None] * self.prob.Xstar, axis = 0)

        bacond = (self.alphas > 1e-5)
        bdcond = (self.deltas + self.C > 1e-5)

        bcond = np.array([a and b for a, b in zip(bacond, bdcond)]).flatten()

        clf = classifier()
        clf.w = self.w
        self.b = self.getB()
        clf.b = self.b
        clf.alphas = np.asarray(self.alphas+self.etas)
        clf.xs = x
        clf.ys = y
        clf.kern = prob.xkernel
        clf.support_vectors = self.x[bacond.flatten()]
        #clf.scaler = prob.scaler

        priv_clf = classifier()
        priv_clf.w = self.wStar
        priv_clf.b = self.getBstar()
        priv_clf.support_vectors = self.prob.Xstar[np.array(bacond).flatten()]
        return clf #, priv_clf

    def F(self, i):
        runningTotal = 0
        for j in range(self.L):
            runningTotal += self.alphas[j] * self.y[j] * self.prob.xkernel(self.x[i], self.x[j])
        return runningTotal[0]

    def f(self, i):
        runningTotal = 0
        for j in range(self.L):
            runningTotal += (self.alphas[j] + self.deltas[j]) * self.prob.xSkernel(self.xStar[i], self.xStar[j])
            if (self.alphas[j] + self.deltas[j] > -1e-5) and (self.alphas[j] + self.deltas[j] < 1e-5):
                print("This makes a < C ",j)
        return runningTotal[0]

    def sPos(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == 1:
                runningTotal += 1-np.dot(self.w, self.prob.X[i])
        return runningTotal

    def sNeg(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == -1:
                runningTotal += -1 - np.dot(self.w, self.prob.X[i])
        return runningTotal

    def nPos(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == 1:
                runningTotal += 1
        return runningTotal if runningTotal > 0 else 1

    def nNeg(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.deltas[i] -self.C < 1e-5 and self.y[i] == -1:
                runningTotal += 1
        return runningTotal if runningTotal > 0 else 1

    def getB(self):
        return ((self.bPlusbStar()/self.nPos())+(self.bMinusbStar()/self.nNeg()))/2

    def getBstar(self):
        return ((self.bPlusbStar()/self.nPos())-(self.bMinusbStar()/self.nNeg()))/2

    def bPlusbStar(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.y[i] == 1:
                #runningTotal += 1 - np.dot(self.wStar, self.xS[i]) - np.dot(self.w, self.x[i])
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.y[j] * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = (1/self.prob.gamma)*abcxx
                runningTotal += 1 - abcxx - ayxx
        return runningTotal

    def bMinusbStar(self):
        runningTotal = 0
        for i in range(self.L):
            if self.alphas[i] > 1e-5 and self.y[i] == -1:
                #runningTotal += -1 + np.dot(self.wStar, self.xS[i]) - np.dot(self.w, self.x[i])
                ayxx = 0
                for j in range(self.prob.num):
                    ayxx += self.alphas[j] * self.y[j] * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = 0
                for j in range(self.prob.num):
                    abcxx += (self.alphas[j] + self.deltas[j]) * self.prob.xkernel(self.x[j], self.x[i])
                abcxx = (1/self.prob.gamma)*abcxx
                runningTotal += -1 + abcxx - ayxx
        return runningTotal

class KT():
    def train(self, prob: svm_problem):
        self.x = prob.X
        self.y = prob.Y
        self.C = prob.C
        self.prob = prob

        self.NUM = self.x.shape[0]

        svm = SVM()
        priv_clf = svm.train(prob.Xstar, prob)

        frames = np.zeros((prob.num,len(priv_clf.support_vectors)))
        for i in range(prob.num):
            for j in range(len(priv_clf.support_vectors)):
                frames[i][j] = prob.xkernel((priv_clf.support_vectors[j]),prob.Xstar[i])

        training_pairs = np.zeros((prob.num,len(priv_clf.support_vectors)), dtype=object)
        for i in range(prob.num):
            for j in range(len(priv_clf.support_vectors)):
                training_pairs[i][j] = [prob.X[i], frames[i][j]]
        training_pairs = np.array(training_pairs)

        regr_pairs = np.zeros((len(priv_clf.support_vectors),prob.num), dtype=object)
        for i in range(prob.num):
            for j in range(len(priv_clf.support_vectors)):
                regr_pairs[j][i] = training_pairs[i][j]

        self.models = []
        self.polyFit = []
        for dataSet in regr_pairs:
            self.regr = SVR(kernel='poly')
            xs = []
            ys = []
            for i in range(prob.num):
                xs.append(dataSet[i][0].flatten())
                ys.append(dataSet[i][1])
            xs = np.array(xs)
            ys = np.array(ys)
            self.models.append(self.regr.fit(xs, ys))

        new_xs = []
        new_ys = []
        for i in range(prob.num):
            new_xs.append(self.F(prob.X[i].reshape(1,-1)).flatten())
            new_ys.append(priv_clf.predict(prob.Xstar[i]))
        new_x = np.array(new_xs)
        new_y = np.array(new_ys)
        new_prob = svm_problem(new_x, prob.Xstar, new_y)

        new_svm = SVMdp()
        self.clf = new_svm.train(new_prob)
        self.support_vectors = self.clf.support_vectors
        self.w = self.clf.w
        self.b = self.clf.b

    def F(self, x):
        toReturn = []
        for i in range(len(self.models)):
            #x_ = self.polyFit[i].transform(x)
            toReturn.append(self.models[i].predict(x))
        toReturn = np.array(toReturn)
        return toReturn

    def predict(self, x):
        new_x = np.array(self.F(x.reshape(1,-1)).flatten())
        return self.clf.predict(new_x.T)

