import os, time, random, numpy as np, torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def make_run_dir(prefix='runs'):
    ts = time.strftime('%Y%m%d-%H%M%S')
    d = os.path.join(prefix, ts)
    os.makedirs(d, exist_ok=True)
    return d
