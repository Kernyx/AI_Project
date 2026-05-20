# Дообучение модели в Google Colab

Цель: загрузить один архив в Google Drive, распаковать его в Colab, дообучить модель на GPU и получить новый файл весов.

## 1. Что загрузить в Google Drive

Не нужно загружать каждую папку вручную. В проекте создается один архив:

```text
tattoo_colab_package.zip
```

Загрузите этот файл в Google Drive в папку:

```text
MyDrive/tattoo_project/
```

## 2. Включить GPU в Colab

В Colab откройте:

```text
Runtime -> Change runtime type -> T4 GPU
```

## 3. Подключить Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 4. Распаковать архив

```python
PROJECT_DIR = "/content/drive/MyDrive/tattoo_project"
ZIP_PATH = f"{PROJECT_DIR}/tattoo_colab_package.zip"

!mkdir -p "$PROJECT_DIR"
!unzip -q -o "$ZIP_PATH" -d "$PROJECT_DIR"
```

После распаковки структура должна быть такой:

```text
tattoo_project/
  NN/
    resnet.py
    data/
  models/
    tattoo_embedding.pth
```

## 5. Проверить GPU и данные

```python
import torch
from pathlib import Path

PROJECT_DIR = "/content/drive/MyDrive/tattoo_project"
DATA_DIR = f"{PROJECT_DIR}/NN/data"
RESUME = f"{PROJECT_DIR}/models/tattoo_embedding.pth"
BEST = f"{PROJECT_DIR}/models/tattoo_embedding.pth"
LATEST = f"{PROJECT_DIR}/models/tattoo_embedding_latest.pth"
METRICS_CSV = f"{PROJECT_DIR}/models/training_metrics.csv"
METRICS_PLOT = f"{PROJECT_DIR}/models/distance_horns.png"

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("classes:", len([p for p in Path(DATA_DIR).iterdir() if p.is_dir()]))
print("resume exists:", Path(RESUME).exists())
```

## 6. Дообучить модель

```python
%cd /content/drive/MyDrive/tattoo_project

!python NN/resnet.py train \
  --data "$DATA_DIR" \
  --resume "$RESUME" \
  --save "$BEST" \
  --latest "$LATEST" \
  --epochs 25 \
  --batch-size 16 \
  --lr 0.00003 \
  --margin 0.2 \
  --num-workers 2 \
  --metrics-csv "$METRICS_CSV" \
  --metrics-plot "$METRICS_PLOT"
```

Если Colab падает по памяти, уменьшите batch size:

```python
!python NN/resnet.py train \
  --data "$DATA_DIR" \
  --resume "$RESUME" \
  --save "$BEST" \
  --latest "$LATEST" \
  --epochs 25 \
  --batch-size 8 \
  --lr 0.00003 \
  --margin 0.2 \
  --num-workers 2 \
  --metrics-csv "$METRICS_CSV" \
  --metrics-plot "$METRICS_PLOT"
```

`--epochs 25` при `--resume` означает еще 25 эпох дообучения от текущего checkpoint.

## 7. Проверить метрики расстояний

После обучения рядом с весами появятся:

```text
models/training_metrics.csv
models/distance_horns.png
```

Для презентации используйте `distance_horns.png`.

Интерпретация:

- `dist(A,P)` - расстояние между фото одного человека, должно снижаться.
- `dist(A,N)` - расстояние между фото разных людей, должно расти.
- Хороший признак - линии расходятся, то есть `dist(A,N) - dist(A,P)` увеличивается.
- Плохой признак - обе линии падают: модель сжимает все в одну точку.
- Плохой признак - обе линии растут: эмбеддинги расходятся хаотично.

## 8. Забрать веса в backend

После обучения основной файл весов уже будет называться правильно:

```text
models/tattoo_embedding.pth
```

Скачайте его из Google Drive и положите в локальный проект с тем же именем:

```bash
cp /path/to/downloaded/tattoo_embedding.pth models/tattoo_embedding.pth
```

И перезапустите backend.

## Важные замечания

- `models/tattoo_embedding.pth` перезаписывается только если loss улучшился.
- `latest` сохраняется после каждой эпохи.
- CSV и PNG с метриками сохраняются в `models/`.
- Для маленького датасета не ставьте слишком большой learning rate. Начинайте с `0.00003`.
- Если добавили новые классы, лучше обучать 20-40 эпох и проверять поиск вручную на отложенных фото.
