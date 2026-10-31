
import numpy as np


class MultilayerPerceptronClassifier:
    def __init__(self, X, D, learning_rate, precision, max_epochs, topology):
        self.X = X
        self.D = D

        self.p, self.N = X.shape
        self.m = D.shape[0]

        self.eta = learning_rate
        self.pr = precision
        self.me = max_epochs

        # Adiciona a camada de saída à topologia
        self.topology = topology + [self.m]

        # Inicialização dos pesos
        self.W = []
        entradas = self.p

        for neuronios in self.topology:
            limite = np.sqrt(6 / (entradas + neuronios))

            pesos = np.random.uniform(
                -limite,
                limite,
                (neuronios, entradas + 1)
            )

            self.W.append(pesos)
            entradas = neuronios

        self.historico_EQM = []

    # Função de ativação
    def g(self, u):
        return (1 - np.exp(-u)) / (1 + np.exp(-u))

    # Derivada da função de ativação
    def g_d(self, y):
        return 0.5 * (1 - y ** 2)

    # Propagação direta
    def forward(self, x):
        a = np.vstack((-np.ones((1, 1)), x))

        ativacoes = []
        entradas = [a]

        for indice, pesos in enumerate(self.W):
            u = pesos @ a
            y = self.g(u)

            ativacoes.append(y)

            if indice < len(self.W) - 1:
                a = np.vstack((-np.ones((1, 1)), y))
                entradas.append(a)

        return ativacoes[-1], ativacoes, entradas

    # Retropropagação
    def backward(self, d, ativacoes, entradas):
        deltas = [None] * len(self.W)

        # Erro da camada de saída
        erro = d - ativacoes[-1]
        deltas[-1] = erro * self.g_d(ativacoes[-1])

        # Erros das camadas ocultas
        for indice in range(len(self.W) - 2, -1, -1):
            pesos_proxima = self.W[indice + 1][:, 1:]

            deltas[indice] = (
                pesos_proxima.T @ deltas[indice + 1]
            ) * self.g_d(ativacoes[indice])

        # Atualização dos pesos
        for indice in range(len(self.W)):
            self.W[indice] += (
                self.eta * deltas[indice] @ entradas[indice].T
            )

    # Cálculo do EQM
    def calcular_EQM(self):
        saidas = self.predict(self.X)
        E = self.D - saidas

        return np.sum(E ** 2) / (2 * self.N * self.m)

    # Treinamento
    def fit(self):
        epoca = 0

        EQM = self.calcular_EQM()
        self.historico_EQM.append(EQM)

        while epoca < self.me and EQM > self.pr:
            indices = np.random.permutation(self.N)

            for k in indices:
                x_k = self.X[:, k].reshape(self.p, 1)
                d_k = self.D[:, k].reshape(self.m, 1)

                _, ativacoes, entradas = self.forward(x_k)

                self.backward(d_k, ativacoes, entradas)

            EQM = self.calcular_EQM()

            if not np.isfinite(EQM):
                raise ValueError(
                    "O EQM divergiu. Verifique a taxa de aprendizagem."
                )

            self.historico_EQM.append(EQM)
            epoca += 1

        return self.W, epoca

    # Predição
    def predict(self, X):
        quantidade = X.shape[1]
        saidas = np.zeros((self.m, quantidade))

        for k in range(quantidade):
            x_k = X[:, k].reshape(self.p, 1)
            y, _, _ = self.forward(x_k)

            saidas[:, k] = y.ravel()

        return saidas

    # Acurácia
    def acuracia(self, X, D):
        saidas = self.predict(X)

        classes_previstas = np.argmax(saidas, axis=0)
        classes_reais = np.argmax(D, axis=0)

        return np.mean(classes_previstas == classes_reais)
