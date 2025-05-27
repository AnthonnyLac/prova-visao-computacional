# classificacao_cifar_local_completo.py

"""
Prova Final A2 - Parte 2: Classificação com CNN (Gato x Cachorro) usando CIFAR-10

"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

# ---------------- Configurações ---------------- #

IMG_SIZE = (32, 32)
LOCAL_IMG_DIR = 'imagens'
EPOCHS = 70
class_names = ['gato', 'cachorro']

# ---------------- Funções Auxiliares ---------------- #

def filtrar_cifar10(x, y, classes=[3,5]):
    """
    Filtra CIFAR-10 para manter apenas as classes desejadas.
    Remapeia: Gato (3) -> 0, Cachorro (5) -> 1.
    """
    idx = np.isin(y.flatten(), classes)
    x_filtered = x[idx]
    y_filtered = y[idx]
    y_mapped = np.array([0 if label == 3 else 1 for label in y_filtered.flatten()])
    return x_filtered, y_mapped

def construir_modelo(input_shape, num_classes):
    """
    Constrói uma CNN simples com camadas convolucionais, pooling e dropout.
    Técnicas de data augmentation aplicadas.
    """
    model = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def carregar_imagem_local(caminho, tamanho=IMG_SIZE):
    """
    Carrega e prepara uma imagem local para classificação.
    """
    img = cv2.imread(caminho)
    img = cv2.resize(img, tamanho)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Formato batch (1, 32, 32, 3)
    return img

# ---------------- Etapa 1: Preparar CIFAR-10 ---------------- #

start_time = time.time()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Filtrando CIFAR-10 para classes gato e cachorro...")

# Filtra e junta os dados
x_data, y_data = filtrar_cifar10(
    np.concatenate([x_train, x_test]), 
    np.concatenate([y_train, y_test])
)

# Normalização
x_data = x_data.astype('float32') / 255.0

# Divisão em 80% treino / 20% teste
x_train_filt, x_test_filt, y_train_filt, y_test_filt = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"Treinamento: {x_train_filt.shape}, Teste: {x_test_filt.shape}")

# ---------------- Etapa 2: Treinar CNN ---------------- #

model = construir_modelo(input_shape=(32,32,3), num_classes=2)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Iniciando treinamento...")
history = model.fit(
    x_train_filt, y_train_filt, 
    epochs=EPOCHS, 
    validation_data=(x_test_filt, y_test_filt),
    verbose=1
)

# ---------------- Etapa 3: Avaliação ---------------- #

# Acurácia no conjunto de teste
test_loss, test_acc = model.evaluate(x_test_filt, y_test_filt, verbose=0)
print(f'\nAcurácia no conjunto de teste: {test_acc:.4f}')

# Previsões no conjunto de teste
y_pred_probs = model.predict(x_test_filt)
y_pred = np.argmax(y_pred_probs, axis=1)

# Relatório de classificação
print("\nRelatório de Classificação (CIFAR-10 - Gato vs Cachorro):")
print(classification_report(y_test_filt, y_pred, target_names=class_names))

# Matriz de confusão
cm = confusion_matrix(y_test_filt, y_pred)
print("Matriz de Confusão:")
print(cm)

# Visualização da matriz de confusão
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - CIFAR-10 (Gato vs Cachorro)')
plt.show()

# ---------------- Etapa 4: Classificar Imagens Locais ---------------- #

print("\nClassificando imagens locais:")

for prefix in class_names:
    for i in range(7):  # imagens de 0 a 6
        filename = f"{prefix}{i}.jpg"
        caminho = os.path.join(LOCAL_IMG_DIR, filename)

        if not os.path.exists(caminho):
            print(f"Imagem não encontrada: {caminho}")
            continue

        img = carregar_imagem_local(caminho)
        pred = model.predict(img)
        pred_class = np.argmax(pred)

        print(f"Imagem: {filename} -> Predito: {class_names[pred_class]} (Confiança: {np.max(pred):.2f})")

# ---------------- Etapa 5: Visualização das Curvas de Aprendizado ---------------- #

plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Desempenho da CNN (Gato vs Cachorro)')
plt.legend()
plt.show()

# ---------------- Tempo Total Gasto ---------------- #

end_time = time.time()
tempo_total = end_time - start_time
print(f"\nTempo total gasto: {tempo_total:.2f} segundos (~ {tempo_total/60:.2f} minutos)")
