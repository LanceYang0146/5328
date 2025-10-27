import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# ===== CIFAR 的标准化参数 =====
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


class NpzImageDataset(Dataset):
    """
    支持扁平化输入，自动检测数据集类型。
    - 对 CIFAR 启用标准化（RGB）
    - 对 FashionMNIST 保持灰度标准化（0~1）
    - 训练集可选数据增强
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
            if x.size == 28*28:          # FMNIST
                x = x.reshape(28, 28)
            elif x.size == 32*32*3:      # CIFAR
                x = x.reshape(32, 32, 3)
            else:
                raise ValueError(f"Unexpected flattened input size: {x.size}")
        return x

    # ---------- 数据增强 ----------
    def _augment_gray(self, x):
        # 灰度图随机水平翻转 + pad + 随机裁剪
        if self.rng.rand() < 0.5:
            x = np.fliplr(x)
        xpad = np.pad(x, ((2, 2), (2, 2)), mode='reflect')
        i = self.rng.randint(0, 5)
        j = self.rng.randint(0, 5)
        return xpad[i:i + 28, j:j + 28]

    def _augment_rgb(self, x):
        # RGB 图随机水平翻转 + pad + 随机裁剪
        if self.rng.rand() < 0.5:
            x = np.ascontiguousarray(np.flip(x, axis=1))
        xpad = np.pad(x, ((4, 4), (4, 4), (0, 0)), mode='reflect')
        i = self.rng.randint(0, 9)
        j = self.rng.randint(0, 9)
        return xpad[i:i + 32, j:j + 32, :]

    def __getitem__(self, idx):
        x = self._maybe_unflatten(self.X[idx])
        y = int(self.y[idx])

        # ---- train 数据增强 ----
        if self.augment:
            if x.ndim == 2:
                x = self._augment_gray(x)
            elif x.ndim == 3 and x.shape[-1] == 3:
                x = self._augment_rgb(x)

        # ---- 转 float + 标准化 ----
        if x.ndim == 2:
            x = x.astype(np.float32) / 255.0
            x = x[None, :, :]  # (1,H,W)
        elif x.ndim == 3 and x.shape[-1] == 3:
            x = x.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))  # (C,H,W)
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
    自动识别 CIFAR/FMNIST 数据集。
    CIFAR -> 开启数据增强 + RGB 标准化
    FMNIST -> 只归一化灰度
    """
    Xtr, Str, Xts, Yts = load_npz(npz_path)

    # 判断数据类型
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
