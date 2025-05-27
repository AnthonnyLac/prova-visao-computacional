# preprocessamento.py

"""
Prova Final A2 - Parte 1: Pré-processamento de Imagens

"""

import cv2
import os
import matplotlib.pyplot as plt
from typing import Tuple, List

IMG_SIZE = (128, 128)
DATA_DIR = 'imagens'

# ----------- Single Responsibility: Operações atômicas -----------

def carregar_imagem(caminho: str) -> any:
    """Carrega a imagem do disco."""
    print(f"Carregando: {caminho}")
    return cv2.imread(caminho)

def redimensionar(imagem: any, tamanho: Tuple[int, int]) -> any:
    """Redimensiona a imagem."""
    print(" - Redimensionando...")
    return cv2.resize(imagem, tamanho)

def aplicar_gaussiano(imagem: any, ksize: Tuple[int, int] = (5, 5)) -> any:
    """Aplica Filtro Gaussiano."""
    print(" - Aplicando Filtro Gaussiano...")
    return cv2.GaussianBlur(imagem, ksize, 0)

def converter_cinza(imagem: any) -> any:
    """Converte para escala de cinza."""
    print(" - Convertendo para tons de cinza...")
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

def equalizar_histograma(imagem: any) -> any:
    """Equaliza histograma da imagem."""
    print(" - Equalizando histograma...")
    return cv2.equalizeHist(imagem)

# ----------- Open/Closed: Pipeline extensível -----------

def pipeline_preprocessamento(caminho_imagem: str) -> Tuple:
    """
    Executa o pipeline completo de pré-processamento.
    Retorna todas as etapas.
    """
    img = carregar_imagem(caminho_imagem)
    img = redimensionar(img, IMG_SIZE)
    blurred = aplicar_gaussiano(img)
    gray = converter_cinza(blurred)
    equalized = equalizar_histograma(gray)

    return img, blurred, gray, equalized

# ----------- Interface Segregation: Exibição isolada -----------

def mostrar_imagens(imagens: List[Tuple[any, str]], titulo: str) -> None:
    """Mostra imagens lado a lado com matplotlib."""
    fig, axs = plt.subplots(1, len(imagens), figsize=(15, 5))
    for i, (img, nome) in enumerate(imagens):
        if len(img.shape) == 2:
            axs[i].imshow(img, cmap='gray')
        else:
            axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].set_title(nome)
        axs[i].axis('off')
    plt.suptitle(titulo)
    plt.show()

# ----------- Dependency Inversion: Alto nível depende da abstração -----------

def processar_todas_imagens(diretorio: str, prefixos: List[str], indices: range) -> None:
    """Processa e mostra todas as imagens com base nos prefixos e índices."""
    for prefix in prefixos:
        for i in indices:
            filename = f"{prefix}{i}.jpg"
            caminho = os.path.join(diretorio, filename)

            if not os.path.exists(caminho):
                print(f"Imagem não encontrada: {caminho}")
                continue

            print(f"\n==> Processando {filename}")
            original, blurred, gray, equalized = pipeline_preprocessamento(caminho)

            mostrar_imagens([
                (original, 'Original'),
                (blurred, 'Gaussiano'),
                (gray, 'Cinza'),
                (equalized, 'Equalizado')
            ], titulo=f"{prefix.upper()} - {filename}")

# ----------- Execução principal -----------

if __name__ == "__main__":
    prefixos = ['cachorro', 'gato']
    indices = range(7)  # 0 até 6

    processar_todas_imagens(DATA_DIR, prefixos, indices)
