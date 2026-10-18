import numpy as np
import matplotlib.pyplot as plt

def grafico_de_espalhamento(x, d):
    plt.scatter(x[:, 0], x[:, 1], c=d)
    plt.show()

def perceptron_simples(x, d, eta, show_graph = False):
    x = np.insert(x, 0, -1, axis=1)
    w = np.zeros(x.shape[1])

    x1 = None
    if show_graph:
        x1 = np.array([x[:, 1].min(), x[:, 1].max()])

    epoch = 1
    erro = True
    while erro:
        erro = False
        for k, amostra in enumerate(x):
            w_t = w.T
            u = w_t @ amostra
            y = sinal_escada_bipolar(u)

            e = (d[k] - y)
            w = w + eta * e * x[k]

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

def sinal_escada_bipolar(u):
    return 1 if u > 0 else -1