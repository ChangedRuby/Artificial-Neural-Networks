
import numpy as np
import matplotlib.pyplot as plt

from dataset import carregar_dataset
from validacao import dividir_treino_teste
from perceptron_simples import PerceptronSimples
from adaline import Adaline
from mlp import MultilayerPerceptronClassifier


# =====================================================
# PARÂMETROS
# =====================================================

R = 10

# Perceptron
eta_PS = 0.01
max_epocas_PS = 1000

# ADALINE
eta_AD = 0.0001
precisao_AD = 0.1
max_epocas_AD = 500

# MLP
eta_MLP = 0.01
precisao_MLP = 0.01
max_epocas_MLP = 500
topology = [100]


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def calcular_matriz_confusao(modelo, nome_modelo, X, Y):

    if nome_modelo == "MLP":
        saidas = modelo.predict(X)
    else:
        saidas = modelo.W @ X

    previstas = np.argmax(saidas, axis=0)
    reais = np.argmax(Y, axis=0)

    quantidade_classes = Y.shape[0]

    matriz = np.zeros(
        (quantidade_classes, quantidade_classes),
        dtype=int
    )

    for real, previsto in zip(reais, previstas):
        matriz[real, previsto] += 1

    return matriz


def exibir_matriz_confusao(matriz, classes, titulo):

    plt.figure(figsize=(12, 10))

    plt.imshow(matriz, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j, i, matriz[i, j],
                ha="center",
                va="center"
            )

    plt.title(titulo)
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")

    plt.tight_layout()
    plt.show()


def exibir_curva(melhor, pior, nome_modelo, precisao=None):

    plt.figure(figsize=(10, 5))

    historico_melhor = melhor["historico"]
    historico_pior = pior["historico"]

    if nome_modelo == "Perceptron Simples":
        plt.plot(
            historico_melhor,
            label=f'Melhor rodada ({melhor["numero"]})'
        )

        plt.plot(
            historico_pior,
            label=f'Pior rodada ({pior["numero"]})'
        )

        plt.ylabel("Quantidade de erros")

    else:
        plt.plot(
            historico_melhor,
            label=f'Melhor rodada ({melhor["numero"]})'
        )

        plt.plot(
            historico_pior,
            label=f'Pior rodada ({pior["numero"]})'
        )

        plt.axhline(
            y=precisao,
            color="red",
            linestyle="--",
            label="Precisão desejada"
        )

        plt.ylabel("EQM")

    plt.xlabel("Épocas")
    plt.title(f"Curvas de aprendizagem - {nome_modelo}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# =====================================================
# CARREGAMENTO DO DATASET
# =====================================================

X, Y, classes = carregar_dataset()

# X possui uma linha de viés para PS e ADALINE
# A MLP adiciona o viés internamente


# =====================================================
# CONFIGURAÇÃO DOS MODELOS
# =====================================================

resultados = {
    "Perceptron Simples": {
        "acuracias": [],
        "melhor": None,
        "pior": None,
        "melhor_acuracia": -1,
        "pior_acuracia": 2
    },

    "ADALINE": {
        "acuracias": [],
        "melhor": None,
        "pior": None,
        "melhor_acuracia": -1,
        "pior_acuracia": 2
    },

    "MLP": {
        "acuracias": [],
        "melhor": None,
        "pior": None,
        "melhor_acuracia": -1,
        "pior_acuracia": 2
    }
}


# =====================================================
# MONTE CARLO
# =====================================================

for rodada in range(R):

    print(f"\n========== RODADA {rodada + 1}/{R} ==========")

    # Uma única divisão para os três modelos.
    # Assim, todos são avaliados nas mesmas amostras.

    X_treino, Y_treino, X_teste, Y_teste = (
        dividir_treino_teste(X, Y)
    )

    # -------------------------------------------------
    # PERCEPTRON SIMPLES
    # -------------------------------------------------

    modelo = PerceptronSimples(
        X_treino,
        Y_treino,
        eta_PS,
        max_epocas_PS
    )

    W, epocas = modelo.fit()

    acuracia_treino = modelo.acuracia(X_treino, Y_treino)
    acuracia_teste = modelo.acuracia(X_teste, Y_teste)

    resultados["Perceptron Simples"]["acuracias"].append(
        acuracia_teste
    )

    print("\n--- PERCEPTRON SIMPLES ---")
    print(f"Épocas: {epocas}")
    print(f"Acurácia de treinamento: {acuracia_treino:.4f}")
    print(f"Acurácia de teste: {acuracia_teste:.4f}")

    dados = {
        "numero": rodada + 1,
        "acuracia": acuracia_teste,
        "W": W.copy(),
        "historico": modelo.historico_erros.copy(),
        "X_teste": X_teste.copy(),
        "Y_teste": Y_teste.copy(),
        "modelo": modelo
    }

    r = resultados["Perceptron Simples"]

    if acuracia_teste > r["melhor_acuracia"]:
        r["melhor_acuracia"] = acuracia_teste
        r["melhor"] = dados

    if acuracia_teste < r["pior_acuracia"]:
        r["pior_acuracia"] = acuracia_teste
        r["pior"] = dados

    # -------------------------------------------------
    # ADALINE
    # -------------------------------------------------

    modelo = Adaline(
        X_treino,
        Y_treino,
        eta_AD,
        precisao_AD,
        max_epocas_AD
    )

    W, epocas = modelo.fit()

    acuracia_treino = modelo.acuracia(X_treino, Y_treino)
    acuracia_teste = modelo.acuracia(X_teste, Y_teste)

    resultados["ADALINE"]["acuracias"].append(
        acuracia_teste
    )

    print("\n--- ADALINE ---")
    print(f"Épocas: {epocas}")
    print(f"EQM final: {modelo.historico_EQM[-1]:.6f}")
    print(f"Acurácia de treinamento: {acuracia_treino:.4f}")
    print(f"Acurácia de teste: {acuracia_teste:.4f}")

    dados = {
        "numero": rodada + 1,
        "acuracia": acuracia_teste,
        "W": W.copy(),
        "historico": modelo.historico_EQM.copy(),
        "X_teste": X_teste.copy(),
        "Y_teste": Y_teste.copy(),
        "modelo": modelo
    }

    r = resultados["ADALINE"]

    if acuracia_teste > r["melhor_acuracia"]:
        r["melhor_acuracia"] = acuracia_teste
        r["melhor"] = dados

    if acuracia_teste < r["pior_acuracia"]:
        r["pior_acuracia"] = acuracia_teste
        r["pior"] = dados

    # -------------------------------------------------
    # MLP
    # -------------------------------------------------

    # Remove o viés, pois a MLP adiciona internamente
    X_treino_MLP = X_treino[1:, :]
    X_teste_MLP = X_teste[1:, :]

    modelo = MultilayerPerceptronClassifier(
        X_treino_MLP,
        Y_treino,
        eta_MLP,
        precisao_MLP,
        max_epocas_MLP,
        topology
    )

    W, epocas = modelo.fit()

    acuracia_treino = modelo.acuracia(
        X_treino_MLP,
        Y_treino
    )

    acuracia_teste = modelo.acuracia(
        X_teste_MLP,
        Y_teste
    )

    resultados["MLP"]["acuracias"].append(
        acuracia_teste
    )

    print("\n--- MLP ---")
    print(f"Épocas: {epocas}")
    print(f"EQM final: {modelo.historico_EQM[-1]:.6f}")
    print(f"Acurácia de treinamento: {acuracia_treino:.4f}")
    print(f"Acurácia de teste: {acuracia_teste:.4f}")

    dados = {
        "numero": rodada + 1,
        "acuracia": acuracia_teste,
        "W": [peso.copy() for peso in W],
        "historico": modelo.historico_EQM.copy(),
        "X_teste": X_teste_MLP.copy(),
        "Y_teste": Y_teste.copy(),
        "modelo": modelo
    }

    r = resultados["MLP"]

    if acuracia_teste > r["melhor_acuracia"]:
        r["melhor_acuracia"] = acuracia_teste
        r["melhor"] = dados

    if acuracia_teste < r["pior_acuracia"]:
        r["pior_acuracia"] = acuracia_teste
        r["pior"] = dados


# =====================================================
# RESULTADOS MONTE CARLO
# =====================================================

print("\n\n========== COMPARAÇÃO DOS MODELOS ==========")

for nome, r in resultados.items():

    acuracias = r["acuracias"]

    print(f"\n--- {nome} ---")
    print(f"Média: {np.mean(acuracias):.4f}")
    print(f"Desvio padrão: {np.std(acuracias):.4f}")
    print(f"Maior acurácia: {np.max(acuracias):.4f}")
    print(f"Menor acurácia: {np.min(acuracias):.4f}")

    print(
        f"Melhor rodada: {r['melhor']['numero']} "
        f"({r['melhor']['acuracia']:.4f})"
    )

    print(
        f"Pior rodada: {r['pior']['numero']} "
        f"({r['pior']['acuracia']:.4f})"
    )


# =====================================================
# GRÁFICOS E MATRIZES DE CONFUSÃO
# =====================================================

for nome, r in resultados.items():

    melhor = r["melhor"]
    pior = r["pior"]

    print(f"\nGerando gráficos e matrizes: {nome}")

    # Curvas de aprendizagem
    if nome == "ADALINE":
        precisao = precisao_AD

    elif nome == "MLP":
        precisao = precisao_MLP

    else:
        precisao = None

    exibir_curva(melhor, pior, nome, precisao)

    # Matrizes de confusão
    matriz_melhor = calcular_matriz_confusao(
        melhor["modelo"],
        nome if nome != "Perceptron Simples" else "Perceptron",
        melhor["X_teste"],
        melhor["Y_teste"]
    )

    matriz_pior = calcular_matriz_confusao(
        pior["modelo"],
        nome if nome != "Perceptron Simples" else "Perceptron",
        pior["X_teste"],
        pior["Y_teste"]
    )

    exibir_matriz_confusao(
        matriz_melhor,
        classes,
        f"Matriz de confusão - {nome} - Melhor rodada ({melhor['numero']})"
    )

    exibir_matriz_confusao(
        matriz_pior,
        classes,
        f"Matriz de confusão - {nome} - Pior rodada ({pior['numero']})"
    )


# =====================================================
# COMPARAÇÃO FINAL DAS ACURÁCIAS
# =====================================================

nomes = list(resultados.keys())

medias = [
    np.mean(resultados[nome]["acuracias"])
    for nome in nomes
]

plt.figure(figsize=(9, 5))

plt.bar(nomes, medias)

plt.ylabel("Acurácia média")
plt.title("Comparação da acurácia média dos modelos")
plt.ylim(0, 1)
plt.grid(axis="y")

for i, valor in enumerate(medias):
    plt.text(
        i,
        valor + 0.01,
        f"{valor:.2%}",
        ha="center"
    )

plt.tight_layout()
plt.show()
