# Нейросеть

В этой папке лежит код обучения модели:

- `resnet.py` - основной training/inference-код.
- `resnet.ipynb` - экспериментальный notebook, не коммитится по умолчанию.
- `data/` - обучающий датасет, не коммитится.
- `tattoo_embedder.pth` - веса после обучения, не коммитятся.

Для запуска backend-приложения скопируйте веса в runtime-папку:

```bash
cp NN/tattoo_embedder.pth models/tattoo_embedding.pth
```

Для дообучения в Google Colab используйте инструкцию `COLAB_TRAINING.md`.
