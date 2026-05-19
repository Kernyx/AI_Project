"""
Tattoo Identification Embedding Network
========================================
Architecture : ResNet-18 (ImageNet pretrained) → FC(512→128) + L2-norm
Loss          : Triplet Loss  (margin = 0.2)
Embedding dim : 128
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ─────────────────────────────────────────────
#  1. Embedding Network
# ─────────────────────────────────────────────

class TattooEmbeddingNet(nn.Module):
    """
    ResNet-18 backbone with a custom projection head.
    Output: L2-normalised 128-dim embedding vector.
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the original FC head; keep everything up to avgpool -> (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)    # (B, 512)
        emb  = self.projector(feat)           # (B, 128)
        return F.normalize(emb, p=2, dim=1)   # L2-norm -> unit sphere


# ─────────────────────────────────────────────
#  2. Triplet Loss
# ─────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    L = max( d(a, p) - d(a, n) + margin, 0 )

    Args:
        margin    : separation margin (default 0.2)
        distance  : 'euclidean' | 'cosine'
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(self, margin: float = 0.2, distance: str = "euclidean",
                 reduction: str = "mean"):
        super().__init__()
        assert distance  in ("euclidean", "cosine")
        assert reduction in ("mean", "sum", "none")
        self.margin    = margin
        self.distance  = distance
        self.reduction = reduction

    def _dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance == "euclidean":
            return (a - b).pow(2).sum(dim=1).clamp(min=0).sqrt()
        return 1.0 - (a * b).sum(dim=1)   # cosine distance

    def forward(self, anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        losses = F.relu(self._dist(anchor, positive)
                        - self._dist(anchor, negative)
                        + self.margin)
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


# ─────────────────────────────────────────────
#  3. Transforms
# ─────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Augmented transform — only for positive pairs during training
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# Clean transform — for inference AND for anchor/negative during training
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Add zero-mean Gaussian noise (applied during training only)."""
    return tensor + torch.randn_like(tensor) * std


# ─────────────────────────────────────────────
#  4. Triplet Dataset
# ─────────────────────────────────────────────

class TattooTripletDataset(Dataset):
    """
    Expected directory structure:
        root/
            person_001/
                img1.jpg
                img2.jpg
            person_002/
                img1.jpg
            ...

    Each __getitem__ returns (anchor, positive, negative):
      - anchor   : INFERENCE_TRANSFORM  (clean, no augmentation)
      - positive : TRAIN_TRANSFORM + Gaussian noise  (same identity)
      - negative : INFERENCE_TRANSFORM               (different identity)

    Args:
        root      : path to the dataset root directory
        noise_std : std of Gaussian noise added to positive pairs
        verify    : if True, validate every file on init and skip broken ones
    """

    def __init__(self, root: str, noise_std: float = 0.02, verify: bool = True):
        self.noise_std = noise_std
        self.samples: list = []              # list of (path, label_idx)
        self.label_to_indices: dict = {}

        skipped   = 0
        label_idx = -1

        for person_dir in sorted(os.listdir(root)):
            dir_path = os.path.join(root, person_dir)
            if not os.path.isdir(dir_path):
                continue
            label_idx += 1

            for fname in os.listdir(dir_path):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                fpath = os.path.join(dir_path, fname)

                if verify and not TattooTripletDataset._is_valid(fpath):
                    print(f"[dataset] WARNING пропущен битый файл: {fpath}")
                    skipped += 1
                    continue

                idx = len(self.samples)
                self.samples.append((fpath, label_idx))
                self.label_to_indices.setdefault(label_idx, []).append(idx)

        # Remove identities with no valid images
        self.label_to_indices = {k: v for k, v in self.label_to_indices.items() if v}
        self.labels = list(self.label_to_indices.keys())

        print(f"[dataset] Загружено: {len(self.samples)} файлов "
              f"| {len(self.labels)} идентификаторов "
              f"| пропущено битых: {skipped}")

    # ── Image validation ──────────────────────────────────────────────────

    @staticmethod
    def _is_valid(path: str) -> bool:
        """
        Two-step validation:
          1. img.verify()  — checks file structure without decoding pixels.
          2. img.load()    — decodes pixel data; catches truncated/broken streams.
        Each step needs a fresh file handle (verify() exhausts the handle).
        """
        try:
            with Image.open(path) as img:
                img.verify()
            # Re-open: verify() closes / invalidates the handle
            with Image.open(path) as img:
                img.load()
            return True
        except Exception:
            return False

    # ── Loading helpers ───────────────────────────────────────────────────

    def _load(self, path: str) -> torch.Tensor:
        """Load without augmentation. Returns black tensor on read error."""
        try:
            with Image.open(path) as img:
                return INFERENCE_TRANSFORM(img.convert("RGB"))
        except Exception as e:
            print(f"[dataset] WARNING ошибка чтения '{path}': {e} -> чёрный тензор")
            return torch.zeros(3, 224, 224)

    def _load_aug(self, path: str) -> torch.Tensor:
        """Load with augmentation + Gaussian noise. Returns black tensor on error."""
        try:
            with Image.open(path) as img:
                t = TRAIN_TRANSFORM(img.convert("RGB"))
                return add_gaussian_noise(t, self.noise_std)
        except Exception as e:
            print(f"[dataset] WARNING ошибка чтения '{path}': {e} -> чёрный тензор")
            return torch.zeros(3, 224, 224)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pick(self, indices: list, exclude_path: str = None) -> str:
        """Random path from index list, trying to avoid exclude_path."""
        attempts = min(len(indices), 10)
        for _ in range(attempts):
            path, _ = self.samples[random.choice(indices)]
            if path != exclude_path:
                return path
        return self.samples[indices[0]][0]   # fallback

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        anchor_path, anchor_label = self.samples[idx]

        # Positive: same identity, augmented
        pos_path = self._pick(self.label_to_indices[anchor_label],
                              exclude_path=anchor_path)

        # Negative: different identity, clean
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        neg_path  = self._pick(self.label_to_indices[neg_label])

        anchor   = self._load(anchor_path)     # clean
        positive = self._load_aug(pos_path)    # augmented
        negative = self._load(neg_path)        # clean

        return anchor, positive, negative


# ─────────────────────────────────────────────
#  5. Training loop
# ─────────────────────────────────────────────

def train(
    data_root: str,
    save_path: str     = "tattoo_embedder.pth",
    epochs: int        = 30,
    batch_size: int    = 16,
    lr: float          = 1e-4,
    margin: float      = 0.2,
    embedding_dim: int = 128,
    device: str        = "auto",
    grad_accum_steps: int = 1,
):
    """
    Full training routine.

    Args:
        data_root     : path to the tattoo dataset (see TattooTripletDataset)
        save_path     : where to write the best model checkpoint
        epochs        : number of training epochs
        batch_size    : mini-batch size (auto-reduced if dataset is small)
        lr            : Adam learning rate
        margin        : Triplet Loss margin
        embedding_dim : output embedding size
        device        : 'auto' | 'cpu' | 'cuda' | 'mps'
        grad_accum_steps : accumulate gradients over N batches before stepping
                           (effectively multiplies batch size, reduces VRAM use)
    """

    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"[train] device = {device}")

    # Recommended env var to reduce CUDA memory fragmentation
    if device == "cuda":
        import os as _os
        _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = TattooTripletDataset(data_root)

    if len(dataset) == 0:
        raise RuntimeError("[train] Датасет пуст — нет валидных изображений.")
    if len(dataset.labels) < 2:
        raise RuntimeError("[train] Нужно минимум 2 идентификатора для Triplet Loss.")

    # Safeguard: don't drop the only batch when dataset is small
    effective_batch = min(batch_size, len(dataset))
    use_drop_last   = len(dataset) >= effective_batch * 2

    if effective_batch != batch_size:
        print(f"[train] WARNING batch_size уменьшен {batch_size}->{effective_batch}")
    if not use_drop_last:
        print("[train] WARNING drop_last=False (маленький датасет)")

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=use_drop_last,
    )
    print(f"[train] {len(dataloader)} батч(ей) / эпоха")

    if len(dataloader) == 0:
        raise RuntimeError(
            f"[train] DataLoader пуст. "
            f"Сэмплов: {len(dataset)}, batch_size: {effective_batch}."
        )

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model     = TattooEmbeddingNet(embedding_dim=embedding_dim).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP scaler — active only on CUDA; on CPU/MPS it's a no-op wrapper
    use_amp = (device == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss = float("inf")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches  = 0

        optimizer.zero_grad()   # zero once before accumulation loop

        for step, (anchor, positive, negative) in enumerate(dataloader):
            anchor   = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            # Forward pass in fp16 when AMP is active
            with torch.cuda.amp.autocast(enabled=use_amp):
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                loss  = criterion(emb_a, emb_p, emb_n)
                # Scale loss for gradient accumulation so the effective
                # gradient magnitude matches a single large-batch step
                loss  = loss / grad_accum_steps

            scaler.scale(loss).backward()

            # Perform optimizer step only every grad_accum_steps batches
            # (or on the last batch of the epoch)
            is_last = (step + 1 == len(dataloader))
            if (step + 1) % grad_accum_steps == 0 or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)     # <- optimizer step
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum_steps  # un-scale for logging
            num_batches  += 1

        scheduler.step()         # <- LR scheduler step (ПОСЛЕ optimizer)

        avg_loss = running_loss / num_batches
        print(f"Epoch [{epoch:>3}/{epochs}]  loss = {avg_loss:.4f}  "
              f"lr = {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss":            best_loss,
                    "embedding_dim":   embedding_dim,
                },
                save_path,
            )
            print(f"  OK Сохранён чекпоинт -> {save_path}  (loss={best_loss:.4f})")

    print(f"\n[train] Готово. Лучший loss = {best_loss:.4f}")
    return model


# ─────────────────────────────────────────────
#  6. Inference utilities
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "auto") -> TattooEmbeddingNet:
    """
    Load a TattooEmbeddingNet from a checkpoint produced by train().
    Returns model in eval() mode on the target device.
    """
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    checkpoint    = torch.load(checkpoint_path, map_location=device)
    embedding_dim = checkpoint.get("embedding_dim", 128)

    model = TattooEmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"[load_model] epoch={checkpoint.get('epoch', '?')} "
          f"loss={checkpoint.get('loss', float('nan')):.4f}  device={device}")
    return model


def extract_embedding(
    image_source,
    model: TattooEmbeddingNet,
    device: str = "auto",
) -> torch.Tensor:
    """
    Extract a 128-dim L2-normalised embedding for a single tattoo image.
    No augmentation is applied (uses clean INFERENCE_TRANSFORM).

    Args:
        image_source : str (file path) | PIL.Image | torch.Tensor (C, H, W)
        model        : loaded TattooEmbeddingNet
        device       : 'auto' | 'cpu' | 'cuda' | 'mps'

    Returns:
        torch.Tensor shape (128,) on CPU
    """
    if device == "auto":
        device = next(model.parameters()).device

    if isinstance(image_source, str):
        with Image.open(image_source) as img:
            tensor = INFERENCE_TRANSFORM(img.convert("RGB"))
    elif isinstance(image_source, Image.Image):
        tensor = INFERENCE_TRANSFORM(image_source.convert("RGB"))
    elif isinstance(image_source, torch.Tensor):
        tensor = image_source
    else:
        raise TypeError(f"Unsupported image type: {type(image_source)}")

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)          # (1, 128)

    return emb.squeeze(0).cpu()      # (128,)


def batch_extract_embeddings(
    image_paths: list,
    model: TattooEmbeddingNet,
    batch_size: int = 32,
    device: str     = "auto",
) -> torch.Tensor:
    """
    Extract embeddings for a list of image paths in batches.
    Unreadable files produce a zero vector and print a warning.

    Returns:
        torch.Tensor shape (N, 128) on CPU
    """
    if device == "auto":
        device = next(model.parameters()).device

    all_embs = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []

        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    tensors.append(INFERENCE_TRANSFORM(img.convert("RGB")))
            except Exception as e:
                print(f"[batch_extract] WARNING пропущен '{p}': {e} -> нулевой вектор")
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            embs = model(batch)
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


def find_most_similar(
    query_emb: torch.Tensor,
    gallery_embs: torch.Tensor,
    top_k: int = 5,
):
    """
    Find the top-k closest embeddings in a gallery by L2 distance.

    Args:
        query_emb    : (128,)
        gallery_embs : (N, 128)
        top_k        : number of results

    Returns:
        distances (top_k,), indices (top_k,) — ascending order
    """
    dists = torch.cdist(query_emb.unsqueeze(0), gallery_embs).squeeze(0)
    k = min(top_k, len(dists))
    return torch.topk(dists, k=k, largest=False)


# ─────────────────────────────────────────────
#  7. CLI entry-point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tattoo Embedding Network")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Обучить модель")
    p_train.add_argument("--data",       required=True)
    p_train.add_argument("--save",       default="tattoo_embedder.pth")
    p_train.add_argument("--epochs",     type=int,   default=30)
    p_train.add_argument("--batch-size", type=int,   default=32)
    p_train.add_argument("--lr",         type=float, default=1e-4)
    p_train.add_argument("--margin",     type=float, default=0.2)

    p_embed = sub.add_parser("embed", help="Извлечь вектор признаков")
    p_embed.add_argument("--image",      required=True)
    p_embed.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            data_root  = args.data,
            save_path  = args.save,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            margin     = args.margin,
        )

    elif args.cmd == "embed":
        net = load_model(args.checkpoint)
        emb = extract_embedding(args.image, net)
        print(f"\nРазмерность: {emb.shape}")
        print(f"Вектор:\n{emb.numpy()}")

    else:
        parser.print_help()