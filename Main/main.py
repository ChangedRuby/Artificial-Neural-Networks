import numpy as np

from dataset import carregar_dataset
from validacao import dividir_treino_teste
from perceptron_simples import PerceptronSimples


# Carregar dataset
X, Y, classes = carregar_dataset()


# Hiperparâmetros
eta = 0.0001
max_epocas = 1000
R = 10


# Armazenar resultados
resultados = []
epocas_rodadas = []
melhor_rodada = None
pior_rodada = None


# Monte Carlo
for rodada in range(R):

    print(f"\nRodada {rodada + 1}/{R}")

    # Dividir treinamento e teste
    X_treino, Y_treino, X_teste, Y_teste = dividir_treino_teste(X, Y)

    # Criar modelo
    modelo = PerceptronSimples(
        X_treino,
        Y_treino,
        eta,
        max_epocas
    )

    # Treinar
    W, epocas = modelo.fit()

    # Avaliar
    acuracia_treino = modelo.acuracia(X_treino, Y_treino)
    acuracia_teste = modelo.acuracia(X_teste, Y_teste)

        # Guardar melhor rodada
    if melhor_rodada is None or acuracia_teste > melhor_rodada["acuracia"]:
        melhor_rodada = {
            "rodada": rodada + 1,
            "acuracia": acuracia_teste,
            "W": W.copy(),
            "epocas": epocas,
            "historico_erros": modelo.historico_erros.copy(),
            "X_teste": X_teste.copy(),
            "Y_teste": Y_teste.copy()
        }

    # Guardar pior rodada
    if pior_rodada is None or acuracia_teste < pior_rodada["acuracia"]:
        pior_rodada = {
            "rodada": rodada + 1,
            "acuracia": acuracia_teste,
            "W": W.copy(),
            "epocas": epocas,
            "historico_erros": modelo.historico_erros.copy(),
            "X_teste": X_teste.copy(),
            "Y_teste": Y_teste.copy()
        }

    # Armazenar
    resultados.append(acuracia_teste)
    epocas_rodadas.append(epocas)

    print("Épocas:", epocas)
    print("Acurácia de treinamento:", acuracia_treino)
    print("Acurácia de teste:", acuracia_teste)


# Estatísticas finais
resultados = np.array(resultados)

media = np.mean(resultados)
desvio = np.std(resultados)
maximo = np.max(resultados)
minimo = np.min(resultados)


print("\n========== RESULTADOS MONTE CARLO ==========")
print("Número de rodadas:", R)
print("Média da acurácia:", media)
print("Desvio padrão:", desvio)
print("Maior acurácia:", maximo)
print("Menor acurácia:", minimo)
print("\n========== MELHOR RODADA ==========")
print("Rodada:", melhor_rodada["rodada"])
print("Acurácia:", melhor_rodada["acuracia"])

print("\n========== PIOR RODADA ==========")
print("Rodada:", pior_rodada["rodada"])
print("Acurácia:", pior_rodada["acuracia"])