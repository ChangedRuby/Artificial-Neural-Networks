from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import helper as help

def sinal(u):
    return 1 if u > 0 else -1

dados = np.loadtxt(Path(__file__).parent.parent / "Datasets" / "spiral_d (1).csv", delimiter=",")
X, d = dados[:, :2], dados[:, 2]

help.grafico_de_espalhamento(X, d)
# help.perceptron_simples(X, d, 0.001, True)
help.adaline(X, d, 0.005, 1e-8, 1000, True)
help.mlp(X, d, 0.005, [2, 2], 1e-3, 1e-8, 1000, True)

print()