# Веса модели

Положите сюда checkpoint для запуска приложения.

Ожидаемый файл по умолчанию:

```text
models/tattoo_embedding.pth
```

Checkpoint должен соответствовать архитектуре из `NN/resnet.py`:

```text
Tattoo Identification Embedding Network
Architecture : ResNet-18 (ImageNet pretrained) -> projector(512->256->128) + L2-norm
Loss          : Triplet Loss (margin = 0.2)
Embedding dim : 128
```

Файлы `.pth`, `.pt` и `.ckpt` игнорируются git, потому что это большие бинарные артефакты.
