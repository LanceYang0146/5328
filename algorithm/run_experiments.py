import argparse, subprocess, sys, json, os, statistics, time, re

# Regular expression to robustly match the "TEST | loss ... acc ..." line
TEST_LINE_RE = re.compile(r'^TEST\s*\|\s*loss\s+([0-9.]+)\s+acc\s+([0-9.]+)\s*$')

def parse_test_acc_from_stdout(stdout: str):
    """
    Parse the test accuracy from the program's standard output.
    We scan from the end because the TEST line usually appears last.
    Returns a float accuracy if found, otherwise None.
    """
    for line in stdout.strip().splitlines()[::-1]:
        m = TEST_LINE_RE.match(line.strip())
        if m:
            try:
                acc = float(m.group(2))
                if 0.0 <= acc <= 1.0:
                    return acc
            except ValueError:
                pass
    return None


def latest_result_json_acc(run_dir_root="runs"):
    """
    Fallback parser: if stdout does not contain a TEST line,
    read the latest runs/<timestamp>/result.json and return test_acc.
    """
    if not os.path.isdir(run_dir_root):
        return None
    subdirs = [os.path.join(run_dir_root, d) for d in os.listdir(run_dir_root)]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: os.path.getmtime(p))
    cand = os.path.join(latest, "result.json")
    if os.path.isfile(cand):
        try:
            with open(cand, "r") as f:
                obj = json.load(f)
            acc = obj.get("test_acc", None)
            if isinstance(acc, (int, float)) and 0.0 <= acc <= 1.0:
                return float(acc)
        except Exception:
            return None
    return None


def main():
    # ----------------- argument parser -----------------
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['FashionMNIST0.3', 'FashionMNIST0.6', 'CIFAR'])
    ap.add_argument('--method', required=True,
                    choices=['forward-knownT', 'forward-estT', 'gce'])
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--data_dir', type=str, default='../data')
    ap.add_argument('--python', type=str, default=sys.executable,
                    help='Python executable to run train.py')
    args = ap.parse_args()

    # Sanity check: make sure we are inside the algorithm/ folder
    if not os.path.isfile('train.py'):
        print("[WARN] train.py not found in current directory. "
              "Please run this script inside the algorithm/ folder.")

    # Generate deterministic seeds for each repetition
    seeds = [1234 + r * 7 for r in range(args.repeats)]
    accs = []       # valid accuracies
    per_run = []    # detailed record for each run

    # ----------------- main experiment loop -----------------
    for seed in seeds:
        cmd = [
            args.python, 'train.py',
            '--dataset', args.dataset,
            '--method', args.method,
            '--epochs', str(args.epochs),
            '--data_dir', args.data_dir,
            '--seed', str(seed),
        ]
        print('Running:', ' '.join(cmd))

        # Capture both stdout and stderr for debugging
        result = subprocess.run(cmd, capture_output=True, text=True)

        # If train.py crashed, log and continue
        if result.returncode != 0:
            print("[ERROR] Subprocess failed (non-zero exit code).")
            print("stdout:\n", result.stdout)
            print("stderr:\n", result.stderr)
            per_run.append({'seed': seed, 'acc': None, 'from': 'failed'})
            continue

        # Try to parse accuracy from stdout
        acc = parse_test_acc_from_stdout(result.stdout)

        # Fallback: read latest runs/<timestamp>/result.json
        if acc is None:
            acc = latest_result_json_acc("runs")
            src = 'result.json'
        else:
            src = 'stdout'

        if acc is None:
            print("[WARN] Could not find TEST accuracy, skipping this run.")
            per_run.append({'seed': seed, 'acc': None, 'from': 'none'})
            continue

        accs.append(acc)
        per_run.append({'seed': seed, 'acc': acc, 'from': src})
        print(f"[OK] seed={seed} acc={acc:.4f} (from {src})")

    # ----------------- statistics & summary -----------------
    valid_accs = [a for a in accs if isinstance(a, (int, float))]
    if len(valid_accs) == 0:
        print("[ERROR] No valid accuracy values collected.")
        return

    mean = sum(valid_accs) / len(valid_accs)
    std = statistics.pstdev(valid_accs) if len(valid_accs) > 1 else 0.0

    print(f'RESULT: dataset={args.dataset} method={args.method} '
          f'repeats={args.repeats} -> mean={mean:.4f} std={std:.4f}')

    summary = {
        'dataset': args.dataset,
        'method': args.method,
        'repeats': args.repeats,
        'epochs': args.epochs,
        'mean': mean,
        'std': std,
        'accs': valid_accs,
        'runs': per_run,  # include seed & source for debugging
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open('summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved summary.json')

# ---------------------------------------------------------
if __name__ == '__main__':
    main()
