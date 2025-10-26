import argparse, os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from datasets import get_loaders, load_npz
from models import make_model
from losses import ForwardCorrectedCELoss, GCELoss
from estimate_T import estimate_T_confident
from utils import set_seed, accuracy, make_run_dir

KNOWN_T = {
    'FashionMNIST0.3': np.array([[0.7, 0.3, 0.0],
                                 [0.0, 0.7, 0.3],
                                 [0.3, 0.0, 0.7]], dtype=np.float32),
    'FashionMNIST0.6': np.array([[0.4, 0.3, 0.3],
                                 [0.3, 0.4, 0.3],
                                 [0.3, 0.3, 0.4]], dtype=np.float32),
}

def build_loss(method, T_tensor=None, q=0.7):
    if method == 'forward-knownT' or method == 'forward-estT':
        assert T_tensor is not None
        return ForwardCorrectedCELoss(T_tensor)
    elif method == 'gce':
        return GCELoss(q=q)
    else:
        raise ValueError(f'Unknown method {method}')

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
        loss = ce(logits, y)  # standard CE for reporting
        losses.append(loss.item())
        accs.append(accuracy(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, choices=['FashionMNIST0.3','FashionMNIST0.6','CIFAR'])
    ap.add_argument('--data_dir', type=str, default='../data')
    ap.add_argument('--method', type=str, required=True, choices=['forward-knownT','forward-estT','gce'])
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--q', type=float, default=0.7, help='GCE q in (0,1]')
    # ---- early stopping hyperparams ----
    ap.add_argument('--patience', type=int, default=5, help='Early stopping patience based on val loss')
    ap.add_argument('--min_delta', type=float, default=1e-4, help='Minimum improvement on val loss to reset patience')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    npz_path = os.path.join(args.data_dir, f'{args.dataset}.npz')
    Xtr, Str, Xts, Yts = load_npz(npz_path)
    is_rgb = (Xts.shape[1:] == (32,32,3))

    tr_loader, val_loader, ts_loader = get_loaders(npz_path, seed=args.seed, batch_size=args.batch_size)

    model = make_model(is_rgb=is_rgb, num_classes=3).to(device)

    # Transition matrix preparation
    T_tensor = None
    if args.method == 'forward-knownT':
        T_tensor = torch.tensor(KNOWN_T[args.dataset], dtype=torch.float32, device=device)
    elif args.method == 'forward-estT':
        # warmup a few epochs with standard CE to obtain probabilities
        warmup_opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ce = nn.CrossEntropyLoss()
        for _ in range(3):
            model.train()
            for x, y in tr_loader:
                x, y = x.to(device), y.to(device)
                warmup_opt.zero_grad()
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()
                warmup_opt.step()
        # estimate T from validation loader for diversity
        T_tensor = estimate_T_confident(model, val_loader, device=device, topk=150, num_classes=3).to(device)

    criterion = build_loss(args.method, T_tensor=T_tensor, q=args.q)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- early stopping states (monitor val_loss) ----
    best_val_loss = float('inf')      # best (lowest) validation loss seen so far
    best_val_acc  = 0.0               # ✅ track best validation accuracy
    best_state = None                 # best model state dict
    epochs_no_improve = 0             # epochs since last improvement
    patience = max(0, args.patience)  # 0 means disable early stopping
    min_delta = float(args.min_delta)

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = one_epoch(model, tr_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f'Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | '
              f'val loss {val_loss:.4f} acc {val_acc:.4f}')

        # save best by val_loss (for generalization)
        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            best_val_acc  = val_acc       # ✅ update best val acc along with best val loss
            best_state = {'model': model.state_dict()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # early stopping check
        if patience > 0 and epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch:02d} '
                  f'(no val loss improvement in {patience} epochs)')
            break

    # load best model (if any)
    if best_state is not None:
        model.load_state_dict(best_state['model'])

    # test
    ts_loss, ts_acc = eval_epoch(model, ts_loader, device)
    print(f'TEST | loss {ts_loss:.4f} acc {ts_acc:.4f}')

    # ---- compute RRE of T_hat if available and true T exists ----
    T_RRE = None
    if T_tensor is not None and args.dataset in KNOWN_T:
        T_true = torch.tensor(KNOWN_T[args.dataset], dtype=torch.float32, device=device)
        num = torch.norm(T_tensor - T_true, p='fro')
        den = torch.norm(T_true, p='fro') + 1e-12
        T_RRE = float((num / den).item())

    # save artifacts
    run_dir = make_run_dir()
    meta = {
        'args': vars(args),
        'method': args.method,
        'best_val_loss': best_val_loss,
        'best_val_acc':  best_val_acc,   # ✅ 保存验证集最佳准确率
        'test_acc':      ts_acc,         # ✅ 测试集准确率
        'test_loss':     ts_loss         # ✅ 测试集损失
    }
    if T_tensor is not None:
        meta['T'] = (T_tensor.detach().cpu().numpy()).tolist()  # ✅ 保存 T-hat
    if T_RRE is not None:
        meta['T_RRE'] = T_RRE                                    # ✅ 保存 T-hat-RRE

    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved run to', run_dir)

if __name__ == '__main__':
    main()
