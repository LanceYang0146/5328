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

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = one_epoch(model, tr_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f'Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}')
        if val_acc > best_val:
            best_val = val_acc
            best_state = { 'model': model.state_dict() }

    # load best
    if 'best_state' in locals():
        model.load_state_dict(best_state['model'])

    # test
    ts_loss, ts_acc = eval_epoch(model, ts_loader, device)
    print(f'TEST | loss {ts_loss:.4f} acc {ts_acc:.4f}')

    # save artifacts
    run_dir = make_run_dir()
    meta = {
        'args': vars(args),
        'test_acc': ts_acc,
        'method': args.method,
    }
    if T_tensor is not None:
        meta['T'] = (T_tensor.detach().cpu().numpy()).tolist()
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved run to', run_dir)

if __name__ == '__main__':
    main()
