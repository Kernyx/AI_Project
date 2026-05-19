# Model Checkpoints

Put the tattoo embedding checkpoint here for local development.

Default expected file:

```text
models/tattoo_embedding.pth
```

The checkpoint must match this architecture:

```text
Tattoo Identification Embedding Network
Architecture : ResNet-18 (ImageNet pretrained) -> FC(512->128) + L2-norm
Loss          : Triplet Loss (margin = 0.2)
Embedding dim : 128
```

Checkpoint files such as `.pth`, `.pt`, and `.ckpt` are ignored by git because they are large binary artifacts.

