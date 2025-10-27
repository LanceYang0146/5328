import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# ===== CIFAR standardization parameters =====
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


class NpzImageDataset(Dataset):
    """
    Supports flattened input and automatically detects dataset type.
    - Enables standardization for CIFAR (RGB)
    - Keeps grayscale normalization (0~1) for FashionMNIST
    - Optional data augmentation for training set
    """
    def __init__(self, X, y, train=False, augment=False, seed=0, standardize_rgb=False):
        self.X = np.asarray(X)
        self.y = np.asarray(y).astype(int)
        self.train = bool(train)
        self.augment = bool(augment) and self.train
        self.standardize_rgb = bool(standardize_rgb)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return int(self.y.shape[0])

    def _maybe_unflatten(self, x):
        if x.ndim == 1:
            if x.size == 28*28:          # FashionMNIST
                x = x.reshape(28, 28)
            elif x.size == 32*32*3:      # CIFAR
                x = x.reshape(32, 32, 3)
            else:
                raise ValueError(f"Unexpected flattened input size: {x.size}")
        return x

    # ---------- Data Augmentation ----------
    def _augment_gray(self, x):
        # Grayscale image: random horizontal flip + padding + random crop
        if self.rng.rand() < 0.5:
            x = np.fliplr(x)
        xpad = np.pad(x, ((2, 2), (2, 2)), mode='reflect')
        i = self.rng.randint(0, 5)
        j = self.rng.randint(0, 5)
        return xpad[i:i + 28, j:j + 28]

    def _augment_rgb(self, x):
        # RGB image: random horizontal flip + padding + random crop
        if self.rng.rand() < 0.5:
            x = np.ascontiguousarray(np.flip(x, axis=1))
        xpad = np.pad(x, ((4, 4), (4, 4), (0, 0)), mode='reflect')
        i = self.rng.randint(0, 9)
        j = self.rng.randint(0, 9)
        return xpad[i:i + 32, j:j + 32, :]

    def __getitem__(self, idx):
        x = self._maybe_unflatten(self.X[idx])
        y = int(self.y[idx])

        # ---- Apply data augmentation for training ----
        if self.augment:
            if x.ndim == 2:
                x = self._augment_gray(x)
            elif x.ndim == 3 and x.shape[-1] == 3:
                x = self._augment_rgb(x)

        # ---- Convert to float + normalization ----
        if x.ndim == 2:
            x = x.astype(np.float32) / 255.0
            x = x[None, :, :]  # (1, H, W)
        elif x.ndim == 3 and x.shape[-1] == 3:
            x = x.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))  # (C, H, W)
            if self.standardize_rgb:
                for c in range(3):
                    x[c] = (x[c] - CIFAR_MEAN[c]) / CIFAR_STD[c]
        else:
            raise ValueError(f"Unexpected image shape: {x.shape}")

        return torch.from_numpy(np.ascontiguousarray(x)), y


def load_npz(path):
    d = np.load(path)
    Xtr, Str = d['Xtr'], d['Str']
    Xts, Yts = d['Xts'], d['Yts']
    return Xtr, Str, Xts, Yts


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def get_loaders(npz_path, seed=0, batch_size=128, num_workers=2):
    """
    Automatically detects CIFAR or FashionMNIST dataset.
    CIFAR -> enable data augmentation + RGB standardization
    FashionMNIST -> grayscale normalization only
    """
    Xtr, Str, Xts, Yts = load_npz(npz_path)

    # Detect dataset type
    sample = Xtr[0]
    is_cifar = (sample.size == 32 * 32 * 3 or (sample.ndim == 3 and sample.shape[-1] == 3))

    idx = np.arange(len(Str))
    tr_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, shuffle=True, stratify=Str
    )

    tr_ds = NpzImageDataset(
        Xtr[tr_idx], Str[tr_idx],
        train=True, augment=is_cifar, seed=seed,
        standardize_rgb=is_cifar
    )
    val_ds = NpzImageDataset(
        Xtr[val_idx], Str[val_idx],
        train=False, augment=False, seed=seed,
        standardize_rgb=is_cifar
    )
    ts_ds = NpzImageDataset(
        Xts, Yts,
        train=False, augment=False, seed=seed,
        standardize_rgb=is_cifar
    )

    g = torch.Generator()
    g.manual_seed(seed)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           worker_init_fn=_seed_worker, generator=g,
                           persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=_seed_worker,
                            persistent_workers=(num_workers > 0))
    ts_loader = DataLoader(ts_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           worker_init_fn=_seed_worker,
                           persistent_workers=(num_workers > 0))

    return tr_loader, val_loader, ts_loader
