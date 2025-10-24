import argparse, subprocess, sys, json, os, statistics, tempfile, shutil, time, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['FashionMNIST0.3','FashionMNIST0.6','CIFAR'])
    ap.add_argument('--method', required=True, choices=['forward-knownT','forward-estT','gce'])
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--data_dir', type=str, default='../data')
    args = ap.parse_args()

    accs = []
    for r in range(args.repeats):
        seed = 1234 + r * 7
        cmd = [sys.executable, 'train.py',
               '--dataset', args.dataset,
               '--method', args.method,
               '--epochs', str(args.epochs),
               '--data_dir', args.data_dir,
               '--seed', str(seed)]
        print('Running:', ' '.join(cmd))
        out = subprocess.check_output(cmd, text=True)
        # parse last TEST acc from stdout
        for line in out.strip().splitlines():
            if line.startswith('TEST |'):
                parts = line.split()
                acc = float(parts[-1])
                accs.append(acc)
                break

    mean = sum(accs) / len(accs)
    std = (statistics.pstdev(accs) if len(accs)>1 else 0.0)
    print(f'RESULT: dataset={args.dataset} method={args.method} repeats={args.repeats} -> mean={mean:.4f} std={std:.4f}')
    # also save to a file
    with open('summary.json','w') as f:
        json.dump({'dataset': args.dataset, 'method': args.method, 'repeats': args.repeats,
                   'mean': mean, 'std': std, 'accs': accs}, f, indent=2)
    print('Saved summary.json')

if __name__ == '__main__':
    main()
