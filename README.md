# prova-visao-computacional

# Descriçao do problema 

Projeto criado visando desenvolver um classificador automatico de imagens de gatos e cachorros usando CNN. 

Etapas:

1. Pré-processamento de Imagens
2. Classificação com IA

# Justificativa das técnicas utilizadas
Pré processamento de imagens:

Para atingir o resultado esperado foi realizado uma série de etapas, onde primeiro localizamos o path de 6 imagens de gatos e cachorros para dentro do código, com isso mandamos cada uma delas para uma pipeline de processamento que irá carregar a imagem,  redimensionar para (128x128), aplicar o filtro gaussiano (responsável por suavizar ruídos), converter para tons de cinzas (necessário pois a isso realça o contraste e a próxima etapa só aceita imagens com um canal), para então aplicarmos a Equalização de histograma que irá melhorar o contraste da imagem ajustando níveis de intensidade de pixels da imagem.

Classificação com IA:
O modelo utilizado foi CNN, um modelo com destaque para reconhecimento de padrões em imagens e vídeos. Para construir o modelo, utilizamos camadas convolucionais, pooling e dropout. Onde as camadas convolucionais são responsáveis por aplicar os filtros para captar bordas, texturas e outros padrões da imagem. Já o pooling serve para diminuir as dimensões da imagem mantendo somente as características mais fortes delas, isso serve para tornar mais rápido o modelo com ele podendo focar nas características mais importantes. Já o dropout serve para desligar neurônios durante o treino e evitar que o modelo memorize  algum caminho, melhorando sua capacidade de generalização.

Foi realizada uma separação do conjunto em 80%  para treino e 20% para teste, isso para que o modelo seja exposto a imagens que nunca viu antes para sua capacidade de generalização ser avaliada.


# Etapas realizadas
Pré-processamento de Imagens:
- Etapa 1 -> Busca dos paths das imagens locais;
- Etapa 2 -> carrega imagem recuperada graças ao path;
- Etapa 3 -> altera seu tamanho para 128x128;
- Etapa 4 -> Aplica o Filtro Gaussiano;
- Etapa 5 -> Convertendo para cinza
- Etapa 6 -> Aplica a Equalização de histograma.
- Etapa 7 -> Exibe todas as imagens.

Classificação com IA:
- Etapa 1 -> Inicia contador de tempo de execuçao 
- Etapa 2 -> Carrega o dataset cifar10
- Etapa 3 -> Realiza o filtro do dataset para apenas gatos e cachorros.
- Etapa 4 -> Divide os dados filtratos em:  80% treinamento, 20% para teste.
- Etapa 5 -> Constrói uma CNN com camadas convolucionais, pooling e dropout.
- Etapa 6 -> Inicia o treino com 70 epocas. Se fez necessario pois  aumentar a diversidade dos dados de treinamento com RandomFlip e RandomRotation fez cair muito a acurácia.
- Etapa 7 -> Plota a acurácia, Precisão, Recall, F1-score.
- Etapa 8 -> Realiza a classificação de imagens locais.
- Etapa 9 -> Plota as curvas de aprendizado.

# Resultados obtidos
Pré-processamento de Imagens:
![image](https://github.com/user-attachments/assets/184a6a5e-7871-41da-9c4b-76192b36141c)
![image](https://github.com/user-attachments/assets/0ab5e3f4-c499-4785-9265-842837ffe28f)
![image](https://github.com/user-attachments/assets/14914f94-c903-4c94-b272-fe4ddcdf6873)
![image](https://github.com/user-attachments/assets/3e0d1491-0b1c-4196-8747-bfe13ab72768)
![image](https://github.com/user-attachments/assets/c397d82c-48a6-4d3d-a7a7-47c61d5a503c)
![image](https://github.com/user-attachments/assets/c81444cd-c452-4d59-ad94-f790a1c515d5)

Classificação com IA:



