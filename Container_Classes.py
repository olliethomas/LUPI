from Kernels import *
import numpy as np
from sklearn.preprocessing import RobustScaler as ss

class svm_problem():
    def __init__(self, X, Xstar, Y, c=1.0, gamma=1.0, delta=1.0, xk=Linear(), xSk=Linear()):
        self.C = c
        self.gamma = gamma
        self.delta = delta
        self.xkernel = xk
        self.xSkernel = xSk

        if(isinstance(X, np.ndarray)):
            self.X = X
        else:
            self.X = np.array(X)
        if(isinstance(Xstar, np.ndarray)):
            self.Xstar = Xstar
        else:
            self.Xstar = np.array(Xstar)
        if(isinstance(Y, np.ndarray)):
            self.Y = Y
            self.Y = np.asarray(self.Y).reshape(-1)
        else:
            self.Y = np.array(Y)
            self.Y = np.asarray(self.Y).reshape(-1)

        #self.X = np.divide(self.X, self.X.max())
        #self.Xstar = np.divide(self.Xstar, self.Xstar.max())


        self.num = len(self.X)
        self.dimensions = len(self.X[0])
        self.xi_xj = self.gram_matrix(self.X, self.X, self.xkernel)
        self.xstari_xstarj = self.gram_matrix(self.Xstar, self.Xstar, self.xSkernel)
        self.yi_yj = self.gram_matrix(self.Y, self.Y, Linear())

    def gram_matrix(self, X1, X2, kern):
        if isinstance(kern, Gaussian):
            #print(kern.getName())
            SQD = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    SQD[i,j] = np.linalg.norm(X1[i]-X2[j])
            sigma = np.median(SQD)
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j], sigma=sigma)
            return K
        else:
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j])
            return K

class svm_problem_tuple():
    def __init__(self, prob_tuple):
        self.C = prob_tuple[5]
        if len(prob_tuple) == 9: # SVM
            self.gamma = 1
            self.delta = 1
            self.xkernel = prob_tuple[6]
            self.xSkernel = Linear()
        elif len(prob_tuple) == 11: # SVM+
            self.gamma = prob_tuple[6]
            self.delta = 1
            self.xkernel = prob_tuple[7]
            self.xSkernel = prob_tuple[8]
        elif len(prob_tuple) == 12: # SVMd+ - sa
            self.gamma = 1
            self.delta = prob_tuple[6]
            self.xkernel = prob_tuple[7]
            self.xSkernel = prob_tuple[8]
        elif len(prob_tuple) == 13: # SVMd+
            self.gamma = prob_tuple[7]
            self.delta = prob_tuple[6]
            self.xkernel = prob_tuple[8]
            self.xSkernel = prob_tuple[9]
        else:
            print("poorly formed problem")

        if(isinstance(prob_tuple[0], np.ndarray)):
            self.X = prob_tuple[0]
        else:
            self.X = np.array(prob_tuple[0])
        if(isinstance(prob_tuple[1], np.ndarray)):
            self.Xstar = prob_tuple[1]
        else:
            self.Xstar = np.array(prob_tuple[1])
        if(isinstance(prob_tuple[2], np.ndarray)):
            self.Y = prob_tuple[2]
            self.Y = np.asarray(self.Y).reshape(-1)
        else:
            self.Y = np.array(prob_tuple[2])
            self.Y = np.asarray(self.Y).reshape(-1)


        self.num = len(self.X)
        self.dimensions = len(self.X[0])
        self.xi_xj = self.gram_matrix(self.X, self.X, self.xkernel)
        self.xstari_xstarj = self.gram_matrix(self.Xstar, self.Xstar, self.xSkernel)
        self.yi_yj = self.gram_matrix(self.Y, self.Y, Linear())

    def gram_matrix(self, X1, X2, kern):
        if isinstance(kern, Gaussian):
            SQD = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    SQD[i,j] = np.linalg.norm(X1[i]-X2[j])
            sigma = np.median(SQD)
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j], sigma=sigma)
            return K
        else:
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j])
            return K

class svm_test():
    def __init__(self, test_x, test_y):
        if(isinstance(test_x, np.ndarray)):
            self.X = test_x
        else:
            self.X = np.array(test_x)
        if(isinstance(test_y, np.ndarray)):
            self.Y = test_y
        else:
            self.Y = np.array(test_y)


class classifier():

    def __init__(self):
        self.w = 0
        self.b = 0
        self.alphas = []
        self.support_vectors = []
        #self.scaler = object()

    def predict(self, x):
        #if isinstance(x, np.ndarray):
        #    x = x.reshape(1,-1)
        #else:
        #    x = np.array(x).reshape(1,-1)
        #xprime = self.scaler.transform(x).ravel()
        return np.sign(np.dot(self.w,x)+self.b)

    def f_star(self, x, y): # This won't make sense now, but we come back to it later
        return y*(np.dot(self.w,x)+self.b)

class svm_u_problem():
    def __init__(self, X, Xstar, XstarStar, Y, C=1.0, gamma=1.0, sigma=1, delta=1.0, xkernel=Linear(), xSkernel=Linear(), xSSkernel=Linear()):
        self.C = C
        self.gamma = gamma
        self.sigma = sigma
        self.delta = delta
        self.xkernel = xkernel
        self.xSkernel = xSkernel
        self.xSSkernel = xSSkernel

        if(isinstance(X, np.ndarray)):
            self.X = X
        else:
            self.X = np.array(X)
        if(isinstance(Xstar, np.ndarray)):
            self.Xstar = Xstar
        else:
            self.Xstar = np.array(Xstar)
        if(isinstance(XstarStar, np.ndarray)):
            self.XstarStar = XstarStar
        else:
            self.XstarStar = np.array(XstarStar)
        if(isinstance(Y, np.ndarray)):
            self.Y = Y
            self.Y = np.asarray(self.Y).reshape(-1)
        else:
            self.Y = np.array(Y)
            self.Y = np.asarray(self.Y).reshape(-1)

        self.num = len(self.X)
        self.dimensions = len(self.X[0])
        self.xi_xj = self.gram_matrix(self.X, self.X, self.xkernel)
        self.xstari_xstarj = self.gram_matrix(self.Xstar, self.Xstar, self.xSkernel)
        self.xstarstari_xstarstarj = self.gram_matrix(self.XstarStar, self.XstarStar, self.xSSkernel)
        self.yi_yj = self.gram_matrix(self.Y, self.Y, Linear())

    def gram_matrix(self, X1, X2, kern):
        if isinstance(kern, Gaussian):
            #print(kern.getName())
            SQD = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    SQD[i,j] = np.linalg.norm(X1[i]-X2[j])
            sigma = np.median(SQD)
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j], sigma=sigma)
            return K
        else:
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j])
            return K

class svm_u_problem_tuple():
    def __init__(self, p):
        self.C = p[5]
        self.gamma = p[7]
        self.sigma = p[8]
        self.delta = p[6]
        self.xkernel = p[9]
        self.xSkernel = p[10]
        self.xSSkernel = p[11]

        if(isinstance(p[0], np.ndarray)):
            self.X = p[0]
        else:
            self.X = np.array(p[0])
        if(isinstance(p[1], np.ndarray)):
            self.Xstar = p[1]
        else:
            self.Xstar = np.array(p[1])
        if(isinstance(p[1], np.ndarray)):
            self.XstarStar = p[1]
        else:
            self.XstarStar = np.array(p[1])
        if(isinstance(p[2], np.ndarray)):
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

    def gram_matrix(self, X1, X2, kern):
        if isinstance(kern, Gaussian):
            #print(kern.getName())
            SQD = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    SQD[i,j] = np.linalg.norm(X1[i]-X2[j])
            sigma = np.median(SQD)
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j], sigma=sigma)
            return K
        else:
            K = np.zeros((len(X1), len(X1)))
            for i in range(len(X1)):
                for j in range(len(X1)):
                    K[i,j] = kern(X1[i], X2[j])
            return K


