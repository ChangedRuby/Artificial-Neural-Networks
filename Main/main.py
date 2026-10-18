import numpy as np
import matplotlib.pyplot as plt
import helper as help

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

help.grafico_de_espalhamento(X, d)
help.perceptron_simples(X, d, 0.1, True)

print()