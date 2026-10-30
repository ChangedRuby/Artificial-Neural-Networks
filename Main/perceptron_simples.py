
import numpy as np


def degrau_bipolar(u):
    return np.where(u >= 0, 1, -1)


class PerceptronSimples:

    def __init__(self, X, D, eta, max_epocas):

        self.X = X
        self.D = D

        self.p, self.N = X.shape
        self.m = D.shape[0]

        self.eta = eta
        self.max_epocas = max_epocas

        self.W = np.random.random_sample((self.m, self.p)) - 0.5

        self.historico_erros = []

    def predict(self, X):

        u = self.W @ X
        return degrau_bipolar(u)

    def fit(self):

        epoca = 0
        erro = True

        while erro and epoca < self.max_epocas:

            erro = False

            indices = np.random.permutation(self.N)

            for k in indices:

                x_k = self.X[:, k].reshape(self.p, 1)
                d_k = self.D[:, k].reshape(self.m, 1)

                u_k = self.W @ x_k

                y_k = degrau_bipolar(u_k)

                e_k = d_k - y_k

                self.W = self.W + self.eta * (e_k @ x_k.T)

                if np.any(e_k != 0):
                    erro = True

            saidas = self.predict(self.X)

            erros = np.sum(
                np.any(saidas != self.D, axis=0)
            )

            self.historico_erros.append(erros)

            epoca += 1

        return self.W, epoca

    def acuracia(self, X, D):

        u = self.W @ X

        classes_previstas = np.argmax(u, axis=0)
        classes_reais = np.argmax(D, axis=0)

        return np.mean(classes_previstas == classes_reais)