import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class NpzImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # ✅ 新增：自动识别扁平化输入
        if x.ndim == 1:
            if x.size == 28 * 28:  # FashionMNIST
                x = x.reshape(28, 28)
            elif x.size == 32 * 32 * 3:  # CIFAR
                x = x.reshape(32, 32, 3)
            else:
                raise ValueError(f"Unexpected flattened input size: {x.size}")

        # 原始逻辑
        if x.ndim == 2:  # 28x28 grayscale -> (1, H, W)
            x = x[None, :, :].astype(np.float32) / 255.0
        elif x.ndim == 3 and x.shape[-1] == 3:  # HWC -> CHW
            x = x.transpose(2, 0, 1).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unexpected image shape: {x.shape}")

        return torch.from_numpy(x), int(y)

def load_npz(path):
    d = np.load(path)
    Xtr, Str = d['Xtr'], d['Str']
    Xts, Yts = d['Xts'], d['Yts']
    return Xtr, Str, Xts, Yts

def get_loaders(npz_path, seed=0, batch_size=128):
    Xtr, Str, Xts, Yts = load_npz(npz_path)

    idx = np.arange(len(Str))
    tr_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, shuffle=True, stratify=Str
    )

    tr_ds = NpzImageDataset(Xtr[tr_idx], Str[tr_idx])
    val_ds = NpzImageDataset(Xtr[val_idx], Str[val_idx])
    ts_ds  = NpzImageDataset(Xts, Yts)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    ts_loader  = DataLoader(ts_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return tr_loader, val_loader, ts_loader
