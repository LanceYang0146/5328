import argparse, os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from datasets import get_loaders, load_npz
from models import make_model
from losses import ForwardCorrectedCELoss, GCELoss
from estimate_T import estimate_T_confident
from utils import set_seed, accuracy, make_run_dir

# ===== 已知转移矩阵 =====
KNOWN_T = {
    'FashionMNIST0.3': np.array([[0.7, 0.3, 0.0],
                                 [0.0, 0.7, 0.3],
                                 [0.3, 0.0, 0.7]], dtype=np.float32),
    'FashionMNIST0.6': np.array([[0.4, 0.3, 0.3],
                                 [0.3, 0.4, 0.3],
                                 [0.3, 0.3, 0.4]], dtype=np.float32),
}

def build_loss(method, T_tensor=None, q=0.7):
    if method in ['forward-knownT', 'forward-estT']:
        assert T_tensor is not None
        return ForwardCorrectedCELoss(T_tensor)
    elif method == 'gce':
        return GCELoss(q=q)
    else:
        raise ValueError(method)

def one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, accs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(accuracy(logits.detach(), y))
    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    losses, accs = [], []
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        losses.append(loss.item())
        accs.append(accuracy(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))

def frobenius_relative_error(T_hat: torch.Tensor, T_true: torch.Tensor) -> float:
    """RRE = ||T - T_hat||_F / ||T||_F"""
    diff = T_true - T_hat
    num = torch.linalg.norm(diff, ord='fro')
    den = torch.linalg.norm(T_true, ord='fro')
    return (num / (den + 1e-12)).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True,
                    choices=['FashionMNIST0.3','FashionMNIST0.6','CIFAR'])
    ap.add_argument('--data_dir', type=str, default='../data')
    ap.add_argument('--method', type=str, required=True,
                    choices=['forward-knownT', 'forward-estT', 'gce'])
    ap.add_argument('--q', type=float, default=0.7)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=128)
    # 优化器超参：建议 CIFAR 用 SGD+cosine
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=0)
    # 早停
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    # 估计 T 强化相关
    ap.add_argument('--warmup_epochs', type=int, default=8)   # 3 -> 8
    ap.add_argument('--topk', type=int, default=500)          # 150 -> 500
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    npz_path = os.path.join(args.data_dir, f'{args.dataset}.npz')
    Xtr, Str, Xts, Yts = load_npz(npz_path)
    is_rgb = (args.dataset == 'CIFAR')

    tr_loader, val_loader, ts_loader = get_loaders(npz_path, seed=args.seed, batch_size=args.batch_size)
    model = make_model(is_rgb=is_rgb, num_classes=3).to(device)

    # ===== 对于Forward,准备转移矩阵 T =====
    T_tensor = None
    T_RRE = None
    if args.method == 'forward-knownT':
        T_tensor = torch.tensor(KNOWN_T[args.dataset], dtype=torch.float32, device=device)
    elif args.method == 'forward-estT':
        # ---- warm-up：更长、更稳 ----
        warmup_opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        ce = nn.CrossEntropyLoss()
        for _ in range(args.warmup_epochs):
            model.train()
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                warmup_opt.zero_grad()
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()
                warmup_opt.step()
        # ---- 估计 T：更大的 topk  + 行归一化在内部做 ----
        T_est = estimate_T_confident(model, val_loader, device=device,
                                     topk=args.topk, num_classes=3)
        T_tensor = T_est.to(device)
        #  如果是已知 T 的数据集，计算 RRE
        if args.dataset in KNOWN_T:
            T_true = torch.tensor(KNOWN_T[args.dataset], dtype=torch.float32, device=device)
            T_RRE = frobenius_relative_error(T_tensor, T_true)

    elif args.method == 'gce':
        pass
    # ===== 正式训练：SGD + Cosine LR（更适合 CIFAR）=====
    criterion = build_loss(args.method, T_tensor=T_tensor, q=args.q)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = one_epoch(model, tr_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f'Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}')

        scheduler.step()

        improved = (best_val_loss - val_loss) > args.min_delta
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {'model': model.state_dict()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch:02d} (no val loss improvement in {args.patience} epochs)')
            break

    if best_state is not None:
        model.load_state_dict(best_state['model'])

    ts_loss, ts_acc = eval_epoch(model, ts_loader, device)
    print(f'TEST | loss {ts_loss:.4f} acc {ts_acc:.4f}')

    # ===== 保存结果（包含 T、best_val_acc，便于 CSV/summary 汇总）=====
    run_dir = make_run_dir()
    meta = {
        'args': vars(args),
        'test_acc': ts_acc,
        'best_val_acc': best_val_acc,
        'method': args.method,
        'seed': args.seed,
    }
    if args.method == 'forward-estT':          # 只在需要时保存 \hat T 和 RRE
        meta['T'] = (T_tensor.detach().cpu().numpy()).tolist()
        if T_RRE is not None:
            meta['T_RRE'] = float(T_RRE)
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved run to', run_dir)

if __name__ == '__main__':
    main()
