"""
Tattoo Identification Embedding Network
========================================
Architecture : ResNet-18 (ImageNet pretrained) → FC(512→128) + L2-norm
Loss          : Triplet Loss  (margin = 0.2)
Embedding dim : 128
"""

import os
import random
from csv import DictWriter
from pathlib import Path

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


def _default_metrics_paths(save_path: str) -> tuple[str, str]:
    checkpoint_path = Path(save_path)
    metrics_dir = checkpoint_path.parent
    return (
        str(metrics_dir / "training_metrics.csv"),
        str(metrics_dir / "distance_horns.png"),
    )


def _append_epoch_metrics(csv_path: str, row: dict[str, float | int]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "loss", "dist_ap", "dist_an", "gap", "lr"]
    needs_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _plot_distance_metrics(csv_path: str, plot_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[metrics] matplotlib не установлен — CSV сохранен, PNG-график пропущен")
        return

    import csv

    epochs: list[int] = []
    dist_ap: list[float] = []
    dist_an: list[float] = []

    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row["epoch"]))
            dist_ap.append(float(row["dist_ap"]))
            dist_an.append(float(row["dist_an"]))

    if not epochs:
        return

    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dist_ap, marker="o", label="dist(A, P): один человек")
    plt.plot(epochs, dist_an, marker="o", label="dist(A, N): разные люди")
    plt.fill_between(epochs, dist_ap, dist_an, alpha=0.12)
    plt.title("Расхождение расстояний эмбеддингов")
    plt.xlabel("Эпоха")
    plt.ylabel("Euclidean distance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


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
    resume_path: str   = None,
    latest_path: str   = None,
    epochs: int        = 30,
    batch_size: int    = 16,
    lr: float          = 1e-4,
    margin: float      = 0.2,
    embedding_dim: int = 128,
    device: str        = "auto",
    grad_accum_steps: int = 1,
    num_workers: int   = 4,
    resume_optimizer: bool = False,
    metrics_csv: str | None = None,
    metrics_plot: str | None = None,
):
    """
    Full training routine.

    Args:
        data_root     : path to the tattoo dataset (see TattooTripletDataset)
        save_path     : where to write the best model checkpoint
        resume_path   : optional checkpoint to continue training from
        latest_path   : optional path where every epoch checkpoint is saved
        epochs        : number of training epochs
        batch_size    : mini-batch size (auto-reduced if dataset is small)
        lr            : Adam learning rate
        margin        : Triplet Loss margin
        embedding_dim : output embedding size
        device        : 'auto' | 'cpu' | 'cuda' | 'mps'
        grad_accum_steps : accumulate gradients over N batches before stepping
                           (effectively multiplies batch size, reduces VRAM use)
        num_workers   : DataLoader workers. Use 0 on restricted environments.
        resume_optimizer : restore optimizer state from resume checkpoint when possible
        metrics_csv   : CSV file with epoch/loss/dist(A,P)/dist(A,N) history
        metrics_plot  : PNG plot for presentation ("distance horns")
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
        num_workers=num_workers,
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
    model     = TattooEmbeddingNet(embedding_dim=embedding_dim, pretrained=(resume_path is None)).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if metrics_csv is None or metrics_plot is None:
        default_csv, default_plot = _default_metrics_paths(save_path)
        metrics_csv = metrics_csv or default_csv
        metrics_plot = metrics_plot or default_plot

    start_epoch = 1
    end_epoch = epochs
    best_loss = float("inf")
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"[train] resume checkpoint не найден: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_embedding_dim = checkpoint.get("embedding_dim", embedding_dim)
        if checkpoint_embedding_dim != embedding_dim:
            raise RuntimeError(
                f"[train] embedding_dim checkpoint={checkpoint_embedding_dim}, "
                f"а текущий embedding_dim={embedding_dim}"
            )

        model.load_state_dict(checkpoint["model_state"])
        if resume_optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

        print(
            f"[train] resume из {resume_path} "
            f"| epoch={checkpoint.get('epoch', '?')} "
            f"| loss={checkpoint.get('loss', float('nan')):.4f}"
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        end_epoch = start_epoch + epochs - 1
        best_loss = float(checkpoint.get("loss", float("inf")))

    # AMP scaler — active only on CUDA; on CPU/MPS it's a no-op wrapper
    use_amp = (device == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running_loss = 0.0
        running_dist_ap = 0.0
        running_dist_an = 0.0
        running_triplets = 0
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

            with torch.no_grad():
                # These metrics show whether embeddings split correctly:
                # same-person distance should fall, different-person distance should rise.
                dist_ap = criterion._dist(emb_a, emb_p)
                dist_an = criterion._dist(emb_a, emb_n)
                batch_size_actual = anchor.size(0)
                running_dist_ap += dist_ap.sum().item()
                running_dist_an += dist_an.sum().item()
                running_triplets += batch_size_actual

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
        avg_dist_ap = running_dist_ap / running_triplets
        avg_dist_an = running_dist_an / running_triplets
        avg_gap = avg_dist_an - avg_dist_ap
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:>3}/{end_epoch}]  loss = {avg_loss:.4f}  "
              f"dist(A,P) = {avg_dist_ap:.4f}  dist(A,N) = {avg_dist_an:.4f}  "
              f"gap = {avg_gap:.4f}  lr = {current_lr:.2e}")

        _append_epoch_metrics(
            metrics_csv,
            {
                "epoch": epoch,
                "loss": round(avg_loss, 8),
                "dist_ap": round(avg_dist_ap, 8),
                "dist_an": round(avg_dist_an, 8),
                "gap": round(avg_gap, 8),
                "lr": current_lr,
            },
        )
        _plot_distance_metrics(metrics_csv, metrics_plot)

        checkpoint_payload = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss":            avg_loss,
            "dist_ap":         avg_dist_ap,
            "dist_an":         avg_dist_an,
            "distance_gap":    avg_gap,
            "embedding_dim":   embedding_dim,
        }

        if latest_path:
            torch.save(checkpoint_payload, latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_payload["loss"] = best_loss
            torch.save(checkpoint_payload, save_path)
            print(f"  OK Сохранён чекпоинт -> {save_path}  (loss={best_loss:.4f})")

    print(f"\n[train] Готово. Лучший loss = {best_loss:.4f}")
    print(f"[metrics] CSV: {metrics_csv}")
    print(f"[metrics] PNG: {metrics_plot}")
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
    p_train.add_argument("--resume",     default=None)
    p_train.add_argument("--latest",     default=None)
    p_train.add_argument("--grad-accum-steps", type=int, default=1)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--resume-optimizer", action="store_true")
    p_train.add_argument("--metrics-csv", default=None)
    p_train.add_argument("--metrics-plot", default=None)

    p_embed = sub.add_parser("embed", help="Извлечь вектор признаков")
    p_embed.add_argument("--image",      required=True)
    p_embed.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            data_root  = args.data,
            save_path  = args.save,
            resume_path = args.resume,
            latest_path = args.latest,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            margin     = args.margin,
            grad_accum_steps = args.grad_accum_steps,
            num_workers = args.num_workers,
            resume_optimizer = args.resume_optimizer,
            metrics_csv = args.metrics_csv,
            metrics_plot = args.metrics_plot,
        )

    elif args.cmd == "embed":
        net = load_model(args.checkpoint)
        emb = extract_embedding(args.image, net)
        print(f"\nРазмерность: {emb.shape}")
        print(f"Вектор:\n{emb.numpy()}")

    else:
        parser.print_help()
