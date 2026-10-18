import numpy as np
import matplotlib.pyplot as plt

def sinal(u):
    return 1 if u > 0 else -1

# N = 10 amostras, p = 2 atributos
X = np.array([
    [4.0, 3.5],
    [1.2, 0.8],
    [5.2, 2.8],
    [2.0, 1.5],
    [3.1, 4.6],
    [0.5, 2.3],
    [6.0, 4.1],
    [1.8, 0.4],
    [4.7, 5.3],
    [2.6, 1.1],
])
d = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
X = np.insert(X, 0, -1, axis=1)
w = np.zeros(X.shape[1])

x1 = np.array([X[:, 1].min(), X[:, 1].max()])

u = 0
epoch = 1
erro = True
while erro:
    erro = False
    for k, amostra in enumerate(X):
        w_t = w.T
        u = w_t @ amostra
        y = sinal(u)

        eta = 0.1
        e = (d[k] - y)
        w = w + eta * e * X[k]

        if d[k] != y:
            erro = True

        plt.cla()
        plt.scatter(X[:, 1], X[:, 2], c=d)
        plt.plot(x1, (w[0] - w[1] * x1) / w[2])
        plt.xlim(X[:, 1].min()-1, X[:, 1].max()+1)
        plt.ylim(X[:, 2].min()-1, X[:, 2].max()+1)
        plt.pause(0.1)

    epoch = epoch + 1

x2 = (w[0] - w[1] * x1) / w[2]
plt.plot(x1, x2, 'b')
plt.show()
print()