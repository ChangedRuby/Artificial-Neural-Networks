
import numpy as np


class Adaline:

    def __init__(self, X, D, eta, precisao, max_epocas):

        self.X = X
        self.D = D

        self.p, self.N = X.shape
        self.m = D.shape[0]

        self.eta = eta
        self.precisao = precisao
        self.max_epocas = max_epocas

        # Um vetor de pesos para cada classificador
        self.W = np.random.random_sample((self.m, self.p)) - 0.5

        self.historico_EQM = []

    def predict(self, X):

        # Saída linear dos classificadores
        return self.W @ X

    def calcular_EQM(self):
        U = self.W @ self.X
        E = self.D - U
        EQM = np.sum(E ** 2) / (2 * self.N * self.m)
        return EQM

    def fit(self):
        epoca = 0
        EQM = self.calcular_EQM()
        self.historico_EQM.append(EQM)

        while epoca < self.max_epocas and EQM > self.precisao:
            indices = np.random.permutation(self.N)

            for k in indices:
                x_k = self.X[:, k].reshape(self.p, 1)
                d_k = self.D[:, k].reshape(self.m, 1)

                u_k = self.W @ x_k
                e_k = d_k - u_k

                self.W = self.W + self.eta * (e_k @ x_k.T)

            EQM = self.calcular_EQM()
            self.historico_EQM.append(EQM)
            epoca += 1

        return self.W, epoca

    def acuracia(self, X, D):

        u = self.predict(X)

        classes_previstas = np.argmax(u, axis=0)
        classes_reais = np.argmax(D, axis=0)

        return np.mean(classes_previstas == classes_reais)
