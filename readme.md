# reader-manometer

📟 **Leitura automática de manômetros analógicos** usando **YOLO (detecção)** e **regressão de ângulo**, com conversão para **porcentagem, pressão e volume**.

Projetado para aplicações **industriais, hospitalares e IoT**, eliminando a necessidade de leitura manual.

---

## ✨ Principais recursos

* Detecção do manômetro via **YOLOv8**
* Regressão precisa do **ângulo do ponteiro**
* Conversão de ângulo → porcentagem
* Cálculo de **pressão** e **volume**
* API simples e reutilizável
* Compatível com pipelines de visão computacional

---

## 📦 Instalação

```bash
pip install reader-manometer
```

> ⚠️ O pacote **inclui modelos treinados**.
> Porem você pode fornecer seus próprios arquivos `.pt`.

---

## 🔧 Requisitos

* Python **3.9+**
* PyTorch
* Ultralytics (YOLOv8)
* OpenCV
* NumPy

---

## 📁 Arquivos necessários

Você precisa informar:

* **Modelo YOLO treinado** (`best.pt`)
* **Modelo de regressão de ângulo** (`regressor.pt`)

Exemplo:

```
reader_manometer/runs/detect/train2/weights/best.pt
reader_manometer/regressor.pt
```

---

## 🚀 Uso rápido

### Exemplo completo

```python
from reader_manometer import Manometer, angle_to_percent, get_volume

man = Manometer(
    model="reader_manometer/runs/detect/train2/weights/best.pt",
    regressor="reader_manometer/regressor.pt"
)
'''
retorna uma lista de angulos, muito util quando uma imagem possue mais
de um manometro
'''
angles = man.get_angle(
    filename="./image3.jpeg"
)

if angles:
    print("ângulos:", angles)

    man_pressure = angles[0]
    man_volume = angles[1]

    percent = angle_to_percent(man_pressure)
    print("porcentagem:", round(percent, 2))

    print("pressão:", round(get_volume(percent, 25), 2))

    vol_percent = angle_to_percent(man_volume)
    print("porcentagem volume:", round(vol_percent, 2))

    print("volume:", round(get_volume(vol_percent, 800), 2))
```

---

## 🧠 API

### `Manometer`

Classe principal responsável pela inferência.

```python
Manometer(model: str, regressor: str)
```

**Parâmetros**

* `model`: caminho para o modelo YOLO (`.pt`)
* `regressor`: caminho para o modelo de regressão de ângulo (`.pt`)

---

### `get_angle()`

```python
angles = man.get_angle(filename: str)
```

**Retorno**

```python
[angulo_1, angulo_2]
```

* Valores em **graus**
* Retorna `None` se não detectar o manômetro

---

### `angle_to_percent()`

```python
percent = angle_to_percent(angle)
```

Converte o ângulo do ponteiro em **porcentagem (0–100%)**, considerando a escala do manômetro.

---

### `get_volume()`

```python
value = get_volume(percent, max_value)
```

Usado para calcular:

* Pressão (ex: `25 bar`)
* Volume (ex: `800 L`)

---

## 🏭 Casos de uso

* Monitoramento de oxigênio hospitalar
* Leitura remota de tanques pressurizados
* Automação industrial
* Integração com ESP32, APIs REST e MQTT
* Dashboards e sistemas SCADA

---

## ⚠️ Observações importantes

* O modelo YOLO deve ser **treinado especificamente** para seu tipo de manômetro apesar de ja pussuir uma boa base
* A regressão depende de **imagens bem enquadradas**
* A escala angular precisa estar configurada corretamente no projeto

---

## 🛣️ Roadmap

* [ ] Interface CLI (`reader-manometer image.jpg`)
* [ ] API REST (FastAPI)
* [ ] Suporte a múltiplos manômetros
* [ ] Exportação MQTT / HTTP
* [ ] Dashboard web

---

## 📄 Licença

MIT License

---
