## Passo a passo para o treinamento de novos modelos

### 1 - Treinar modelo de manometro
    
    python yolo/train_yolo.py

### 2 - Criar dataset de regreção

    python regression/regression_dataset.py

### 3 - Treinar modelo de regressão

    python regression/train.py

### 4 - Usar o modelo para obter angulação no ponteiro(needle)

    seguir o exemplo no readme.md


* em dataset/yolo_dataset é necessario conter dentro das pastas images e labels,
as pastas train e val para o treinamento e validação do mesmo
