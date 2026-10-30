
import os
import cv2
import numpy as np


CAMINHO_DATASET = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Datasets",
    "RecFac",
    "RecFac"
)

ALTURA = 30
LARGURA = 30


def carregar_dataset():

    classes = sorted([
        pasta for pasta in os.listdir(CAMINHO_DATASET)
        if os.path.isdir(os.path.join(CAMINHO_DATASET, pasta))
    ])

    amostras = []
    rotulos = []

    for indice_classe, classe in enumerate(classes):

        caminho_classe = os.path.join(CAMINHO_DATASET, classe)

        arquivos = sorted([
            arquivo for arquivo in os.listdir(caminho_classe)
            if arquivo.lower().endswith(".png")
        ])

        if len(arquivos) != 32:
            raise ValueError(
                f"A classe {classe} possui {len(arquivos)} imagens, "
                "mas eram esperadas 32."
            )

        for arquivo in arquivos:

            caminho_imagem = os.path.join(caminho_classe, arquivo)

            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

            if imagem is None:
                raise ValueError(f"Não foi possível ler: {caminho_imagem}")

            imagem = cv2.resize(
                imagem,
                (LARGURA, ALTURA),
                interpolation=cv2.INTER_AREA
            )
            imagem = imagem.astype(np.float64)
            imagem = (imagem / 127.5) - 1

            vetor = imagem.flatten()

            amostras.append(vetor)

            rotulos.append(indice_classe)

    
    X = np.array(amostras).T

    X = np.insert(X, 0, -1, axis=0)


    Y = -np.ones((len(classes), len(rotulos)))

    for indice_amostra, classe in enumerate(rotulos):
        Y[classe, indice_amostra] = 1


    print("Classes encontradas:", len(classes))
    print("Total de imagens:", len(rotulos))
    print("Dimensão de X:", X.shape)
    print("Dimensão de Y:", Y.shape)

    return X, Y, classes
