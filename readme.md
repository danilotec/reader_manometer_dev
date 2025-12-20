# Projeto completo: Leitura de Manômetro com YOLO + Regressão

Este guia te conduz **do zero até a inferência final**, usando **YOLO para detectar o ponteiro** e **uma rede de regressão para estimar o ângulo**.

O foco é **robustez industrial** (iluminação ruim, ângulo da câmera, sujeira).

---

## 1️⃣ Visão geral da arquitetura

Pipeline:

```
Imagem → YOLO → bounding box do ponteiro
                    ↓
            Crop do ponteiro
                    ↓
        CNN de regressão → ângulo (0–360°)
                    ↓
           Conversão para valor físico
```

Decisão importante:

* YOLO **não lê valor**, só localiza o ponteiro
* A regressão **aprende o ângulo**, não regras geométricas

---

## 2️⃣ Ambiente e dependências

### Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate
```

### Instalar dependências

```bash
pip install ultralytics opencv-python torch torchvision numpy matplotlib
```

Teste:

```bash
yolo checks
```

---

## 3️⃣ Estrutura do projeto

```
manometro_ai/
├── data/
│   ├── raw_images/
│   ├── yolo_dataset/
│   │   ├── images/
│   │   └── labels/
│   └── regression_dataset/
├── yolo/
│   ├── train_yolo.py
│   └── gauge.yaml
├── regression/
│   ├── model.py
│   ├── train.py
│   └── infer.py
├── pipeline/
│   └── infer_full.py
└── README.md
```

---

## 4️⃣ Coleta de imagens (FUNDAMENTAL)

📸 Tire **50–200 fotos** do manômetro:

* diferentes valores
* luz forte / fraca
* câmera torta
* reflexo no vidro

Salve em:

```
data/raw_images/
```

👉 Quanto mais variação, melhor o modelo.

---

## 5️⃣ Dataset YOLO (detecção do ponteiro)

### 5.1 Rotulagem

Use **LabelImg** ou **Roboflow**.

Classe única:

```
needle
```

Cada bounding box deve pegar **apenas o ponteiro**, não o centro inteiro.

### 5.2 Estrutura YOLO

```
data/yolo_dataset/
├── images/
│   ├── img1.jpg
│   └── img2.jpg
└── labels/
    ├── img1.txt
    └── img2.txt
```

Formato label:

```
0 x_center y_center width height
```

---

## 6️⃣ Configuração YOLO

### gauge.yaml

```yaml
path: data/yolo_dataset
train: images
val: images

nc: 1
names: ["needle"]
```

### Treino

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="gauge.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

Resultado:

```
runs/detect/train/weights/best.pt
```

---

## 7️⃣ Dataset de regressão (ângulo)

### 7.1 Gerar crops do ponteiro

Use o YOLO treinado para recortar automaticamente:

```python
from ultralytics import YOLO
import cv2, os

yolo = YOLO("best.pt")

for img_name in os.listdir("data/raw_images"):
    img = cv2.imread(f"data/raw_images/{img_name}")
    r = yolo(img)[0]
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"data/regression_dataset/{img_name}", crop)
```

### 7.2 Labels de ângulo

Crie um CSV:

```
image,angle
img1.jpg,45
img2.jpg,132
```

⚠️ Ângulo real medido manualmente (gabarito).

Normalize:

```
angle_norm = angle / 360
```

---

## 8️⃣ Modelo de regressão

### model.py

```python
import torch.nn as nn

class AngleRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

---

## 9️⃣ Treino da regressão

* Entrada: imagem do ponteiro
* Saída: ângulo normalizado
* Loss: MSE

```python
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Treine até erro < 2–3 graus.

---

## 🔟 Pipeline final (inferência completa)

```python
img → YOLO → crop → regressão → ângulo → valor físico
```

Conversão:

```python
valor = escala_min + ang_norm * (escala_max - escala_min)
```

---

## 1️⃣1️⃣ Boas práticas industriais

✔ normalize iluminação
✔ aumente dataset com blur / brilho
✔ use câmera fixa
✔ faça calibração inicial (zero)

---

## 1️⃣2️⃣ Próximos passos

* Converter modelos para ONNX
* Rodar em C++
* Edge AI (Jetson / Coral)
* Detectar números da escala

---

## 🎯 Resultado esperado

Precisão típica:

* ±1–3° de erro
* leitura estável
* robusto a ruído

---

Se quiser, no próximo passo eu posso:

➡️ te ajudar a **rotular imagens corretamente**
➡️ montar o **script de treino da regressão completo**
➡️ adaptar para **C++ / ONNX**
➡️ calibrar para o **seu manômetro específico**
