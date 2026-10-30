import math

import numpy as np
import matplotlib.pyplot as plt

def grafico_de_espalhamento(x, d):
    plt.scatter(x[:, 0], x[:, 1], c=d)
    plt.show()

def perceptron_simples(x, d, eta, show_graph = False):
    x = np.insert(x, 0, -1, axis=1)
    w = np.random.uniform(-0.5, 0.5, x.shape[1])

    x1 = None
    if show_graph:
        x1 = np.array([x[:, 1].min(), x[:, 1].max()])

    epoch = 0
    erro = True
    while erro:
        erro = False
        for k, amostra in enumerate(x):
            w_t = w.T
            u = w_t @ amostra
            y = sinal_escada_bipolar(u)

            e = (d[k] - y)
            w = w + eta * e * amostra

            if d[k] != y:
                erro = True

        if show_graph:
            plt.cla()
            plt.scatter(x[:, 1], x[:, 2], c=d)
            plt.plot(x1, (w[0] - w[1] * x1) / w[2])
            plt.xlim(x[:, 1].min() - 1, x[:, 1].max() + 1)
            plt.ylim(x[:, 2].min() - 1, x[:, 2].max() + 1)
            plt.pause(0.1)

        epoch = epoch + 1

    if show_graph:
        plt.show()
    return w, epoch

def adaline(x, d, eta, convergencia, max_epoch, show_graph = False):
    x = np.insert(x, 0, -1, axis=1)
    w = np.random.uniform(-0.5, 0.5, x.shape[1])

    x1 = None
    if show_graph:
        x1 = np.array([x[:, 1].min(), x[:, 1].max()])

    epoch = 0
    eqm_atual, eqm_anterior = calcular_eqm(x, d, w), float('inf')
    while abs(eqm_atual - eqm_anterior) >= convergencia and epoch < max_epoch:
        eqm_anterior = eqm_atual
        for k, amostra in enumerate(x):
            u = w.T @ amostra
            e = (d[k] - u)
            w = w + eta * e * amostra

            y = sinal_escada_bipolar(u)

        if show_graph and abs(w[2]) > 1e-6:
            plt.cla()
            plt.scatter(x[:, 1], x[:, 2], c=d)
            plt.plot(x1, (w[0] - w[1] * x1) / w[2])
            plt.xlim(x[:, 1].min() - 1, x[:, 1].max() + 1)
            plt.ylim(x[:, 2].min() - 1, x[:, 2].max() + 1)
            plt.pause(0.01)

        eqm_atual = calcular_eqm(x, d, w)
        epoch = epoch + 1

    if show_graph:
        plt.show()
    return w, epoch

def sinal_escada_bipolar(u):
    return 1 if u > 0 else -1

def calcular_eqm(x, d, w):
    e = d - (x @ w)
    return np.sum(e**2)/(2*x.shape[0])

# def calcular_eqm_antigo(x, d, w):
#     eqm = 0
#     for k, amostra in enumerate(x):
#         u = (w.T @ amostra)
#         e = d[k] - u
#         eq = (e**2)/2
#         eqm = eqm + eq
#     eqm = eqm/x.shape[0]
#     return eqm

def mlp_prever(x, w):
    for wi in w:
        x = np.tanh(np.insert(x, 0, -1, axis=1) @ wi)
    return x.ravel()

def mlp(x, d, eta, camadas, tolerancia, convergencia, max_epoch, show_graph = False):
    # Inicializacao dos pesos: w[i] tem uma linha a mais para o bias
    d = d.reshape(-1, 1)
    entradas =[x.shape[1]] + list(camadas)
    saidas = list(camadas) + [1]
    w = [np.random.uniform(-1, 1, (n + 1, m)) for n, m in zip(entradas, saidas)]

    if show_graph:
        g1, g2 = np.meshgrid(np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 50),
                             np.linspace(x[:, 1].min() - 1, x[:, 1].max() + 1, 50))
        grade = np.c_[g1.ravel(), g2.ravel()]

    eqm_anterior = float('inf')
    for epoch in range(max_epoch):
        # Forward: a[i] e a entrada (com bias) da camada i, y e a saida da rede
        a = [np.insert(x, 0, -1, axis=1)]
        for wi in w[:-1]:
            a.append(np.insert(np.tanh(a[-1] @ wi), 0, -1, axis=1))
        y = np.tanh(a[-1] @ w[-1])

        # Erro e criterios de parada
        eqm = np.mean((d - y)**2) / 2
        if eqm < tolerancia or abs(eqm_anterior - eqm) < convergencia:
            break
        eqm_anterior = eqm

        # Backpropagation: do erro da saida ate a primeira camada
        delta = (d - y) * (1 - y**2)
        for i in reversed(range(len(w))):
            grad = a[i].T @ delta / len(x)
            if i > 0:
                delta = (delta @ w[i][1:].T) * (1 - a[i][:, 1:]**2)
            w[i] = w[i] + eta * grad

        # Grafico
        if show_graph:
            plt.cla()
            plt.contourf(g1, g2, mlp_prever(grade, w).reshape(g1.shape), levels=[-1, 0, 1], alpha=0.3)
            plt.scatter(x[:, 0], x[:, 1], c=d.ravel())
            plt.pause(0.01)

    if show_graph:
        plt.show()
    return w, epoch
