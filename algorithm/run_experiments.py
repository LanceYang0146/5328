import argparse, subprocess, sys, json, os, statistics, time, re, csv

# Robust regex to match: "TEST | loss <num> acc <num>"
TEST_LINE_RE = re.compile(r'^TEST\s*\|\s*loss\s+([0-9.]+)\s+acc\s+([0-9.]+)\s*$')

def parse_test_acc_from_stdout(stdout: str):
    """
    Parse test accuracy from stdout by matching the 'TEST | loss ... acc ...' line.
    Return a float in [0,1] or None if not found.
    """
    for line in stdout.strip().splitlines()[::-1]:  # scan from the end
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
    Fallback: read the latest runs/<timestamp>/result.json and return 'test_acc'.
    Return float in [0,1] or None if not available.
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

def append_row_to_csv(csv_path, header, row):
    """
    Append one row to CSV; create the file with header if it doesn't exist.
    """
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['FashionMNIST0.3','FashionMNIST0.6','CIFAR'])
    parser.add_argument('--method', required=True,
                        choices=['forward-knownT','forward-estT','gce'])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--csv', type=str, default='results.csv',
                        help='Path to CSV file for appending results')
    args = parser.parse_args()

    if not os.path.isfile('train.py'):
        print("[WARN] train.py not found in current directory. Run inside algorithm/")

    seeds = [1234 + r * 7 for r in range(args.repeats)]
    accs = []
    header = ["dataset", "method", "seed", "epochs", "test_acc", "time"]

    for seed in seeds:
        cmd = [
            args.python, 'train.py',
            '--dataset', args.dataset,
            '--method', args.method,
            '--epochs', str(args.epochs),
            '--data_dir', args.data_dir,
            '--seed', str(seed),
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("[ERROR] Subprocess failed. Skipping this run.")
            print("stdout:\n", result.stdout)
            print("stderr:\n", result.stderr)
            continue

        # Prefer stdout; fallback to runs/<ts>/result.json
        acc = parse_test_acc_from_stdout(result.stdout)
        if acc is None:
            acc = latest_result_json_acc("runs")
        if acc is None:
            print("[WARN] TEST accuracy not found. Skipping this run.")
            continue

        accs.append(acc)
        append_row_to_csv(
            args.csv, header,
            [args.dataset, args.method, seed, args.epochs, f"{acc:.4f}", time.strftime('%Y-%m-%d %H:%M:%S')]
        )
        print(f"[OK] seed={seed} acc={acc:.4f} -> appended to {args.csv}")

    if not accs:
        print("[ERROR] No valid accuracy collected. CSV might not include any rows.")
        return

    mean = sum(accs) / len(accs)
    std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    print(f"RESULT: dataset={args.dataset} method={args.method} repeats={len(accs)} "
          f"-> mean={mean:.4f} std={std:.4f}")

    # Save a machine-readable summary.json (optional but useful)
    with open("summary.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "method": args.method,
            "repeats": len(accs),
            "epochs": args.epochs,
            "mean": mean,
            "std": std,
            "accs": accs,
            "time": time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)

    # Append one final summary row to CSV (seed='ALL', test_acc='mean±std')
    append_row_to_csv(
        args.csv, header,
        [args.dataset, args.method, "ALL", args.epochs, f"{mean:.4f}±{std:.4f}", time.strftime('%Y-%m-%d %H:%M:%S')]
    )
    print(f"[OK] Summary appended to {args.csv}")

if __name__ == "__main__":
    main()
