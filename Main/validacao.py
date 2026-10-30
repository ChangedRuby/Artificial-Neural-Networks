import numpy as np

def dividir_treino_teste(X, Y, proporcao_treino=0.8):
    N = X.shape[1]
    classes = np.argmax(Y, axis=0)

    indices_treino = []
    indices_teste = []

    for classe in np.unique(classes):
        indices_classe = np.where(classes == classe)[0]

        np.random.shuffle(indices_classe)

        quantidade_treino = int(len(indices_classe) * proporcao_treino)

        indices_treino.extend(indices_classe[:quantidade_treino])
        indices_teste.extend(indices_classe[quantidade_treino:])

    np.random.shuffle(indices_treino)
    np.random.shuffle(indices_teste)

    X_treino = X[:, indices_treino]
    Y_treino = Y[:, indices_treino]

    X_teste = X[:, indices_teste]
    Y_teste = Y[:, indices_teste]

    return X_treino, Y_treino, X_teste, Y_teste