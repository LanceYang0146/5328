import argparse, subprocess, sys, json, os, statistics, time, csv

# ---------- helpers ----------
def list_run_dirs(root="runs"):
    if not os.path.isdir(root):
        return set()
    return set(os.path.join(root, d) for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d)))

def read_this_run_result(run_dir):
    p = os.path.join(run_dir, "result.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {
            "test_acc":      obj.get("test_acc", None),
            "best_val_acc":  obj.get("best_val_acc", None),
            "T":             obj.get("T", None),
            "T_RRE":         obj.get("T_RRE", None),
        }
    except Exception:
        return {}

def append_row_tsv(tsv_path, header, row):
    need_header = not os.path.exists(tsv_path)
    with open(tsv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if need_header:
            w.writerow(header)
        w.writerow(row)

# ---------- main ----------
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
    parser.add_argument('--csv', type=str, default='results.csv')
    args = parser.parse_args()

    seeds = [1 + r * 7 for r in range(args.repeats)]

    header = [
        "ts","dataset","model",
        "acc-hat","acc","acc-val-hat","acc-val","acc-clean",
        "T-hat-RRE","T-hat",
        "acc-hat-std","acc-std","acc-val-hat-std","acc-val-std","acc-clean-std",
        "T-hat-RRE-std","T-hat-std"
    ]

    acc_list, val_acc_list, rre_list = [], [], []
    That_list = []

    for seed in seeds:
        before = list_run_dirs("runs")

        cmd = [
            args.python, 'train.py',
            '--dataset', args.dataset,
            '--method',  args.method,
            '--epochs',  str(args.epochs),
            '--data_dir', args.data_dir,
            '--seed',    str(seed),
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        after = list_run_dirs("runs")
        new_dirs = sorted(list(after - before))
        run_dir = new_dirs[-1] if new_dirs else None
        ts_now = time.strftime('%Y-%m-%d %H:%M:%S')

        if result.returncode != 0 or not run_dir:
            print("[WARN] run failed or run_dir missing; writing empty row")
            append_row_tsv(args.csv, header,
                [ts_now, args.dataset, args.method,
                 "", "", "", "", "",
                 "", "",
                 "", "", "", "", "",
                 "", ""]
            )
            continue

        res = read_this_run_result(run_dir)
        test_acc     = res.get("test_acc", None)
        best_val_acc = res.get("best_val_acc", None)
        T_hat        = res.get("T", None)
        T_rre        = res.get("T_RRE", None)

        if isinstance(test_acc, (int, float)):
            acc_list.append(float(test_acc))
        if isinstance(best_val_acc, (int, float)):
            val_acc_list.append(float(best_val_acc))
        if isinstance(T_rre, (int, float)):
            rre_list.append(float(T_rre))
        if isinstance(T_hat, list):
            That_list.append(T_hat)

        row = [
            ts_now, args.dataset, args.method,
            f"{test_acc:.4f}" if test_acc is not None else "",
            f"{test_acc:.4f}" if test_acc is not None else "",
            f"{best_val_acc:.4f}" if best_val_acc is not None else "",
            f"{best_val_acc:.4f}" if best_val_acc is not None else "",
            "",
            f"{T_rre:.6f}" if T_rre is not None else "",
            json.dumps(T_hat) if T_hat is not None else "",
            "", "", "", "", "", "", ""
        ]
        append_row_tsv(args.csv, header, row)
        print(f"[OK] seed={seed} appended.")

    def mean_std(lst):
        if not lst:
            return 0.0, 0.0
        m = sum(lst)/len(lst)
        s = statistics.pstdev(lst) if len(lst) > 1 else 0.0
        return m, s

    acc_mean, acc_std = mean_std(acc_list)
    val_mean, val_std = mean_std(val_acc_list)
    rre_mean, rre_std = mean_std(rre_list)

    That_std_str = ""
    T_mean = None
    if That_list:
        try:
            import numpy as np
            mats = [np.array(T) for T in That_list]
            shapes = {M.shape for M in mats}
            if len(shapes) == 1:
                A = np.stack(mats, axis=0)
                mu = A.mean(axis=0)
                T_mean = mu.tolist()
                diffs = A - mu
                fro_each = np.linalg.norm(diffs.reshape(len(mats), -1), axis=1)
                That_std_str = f"{fro_each.mean():.6f}"
        except Exception:
            That_std_str = ""

    ts_now = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_row = [
        ts_now, args.dataset, args.method,
        f"{acc_mean:.4f}", f"{acc_mean:.4f}",
        f"{val_mean:.4f}", f"{val_mean:.4f}",
        "",
        f"{rre_mean:.6f}" if rre_list else "",
        "",
        f"{acc_std:.4f}", f"{acc_std:.4f}",
        f"{val_std:.4f}" if val_acc_list else "",
        f"{val_std:.4f}" if val_acc_list else "",
        "",
        f"{rre_std:.6f}" if rre_list else "",
        That_std_str
    ]
    append_row_tsv(args.csv, header, summary_row)
    print(f"[OK] summary appended → {args.csv}")

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "repeats": len(seeds),
        "epochs": args.epochs,
        "mean_acc": acc_mean,
        "std_acc": acc_std,
        "mean_val_acc": val_mean,
        "std_val_acc": val_std,
        "T_mean": T_mean,
        "time": ts_now,
        "mean_rre": rre_mean if rre_list else None,
        "std_rre": rre_std if rre_list else None,
    }
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[OK] summary.json saved with T_mean")

if __name__ == "__main__":
    main()
