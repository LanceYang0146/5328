import argparse, subprocess, sys, json, os, statistics, time, csv

# ---------- helpers ----------
def list_run_dirs(root="runs"):
    if not os.path.isdir(root):
        return set()
    return set(os.path.join(root, d) for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d)))

def read_this_run_result(run_dir):
    """Read fields from this run's runs/<ts>/result.json; return dict."""
    p = os.path.join(run_dir, "result.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r") as f:
            obj = json.load(f)
        return {
            "test_acc":      obj.get("test_acc", None),
            "best_val_acc":  obj.get("best_val_acc", None),
            "T":             obj.get("T", None),       # list[list[...]]
            "T_RRE":         obj.get("T_RRE", None),   # float
        }
    except Exception:
        return {}

def append_row_tsv(tsv_path, header, row):
    need_header = not os.path.exists(tsv_path)
    with open(tsv_path, "a", newline="") as f:
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
    parser.add_argument('--csv', type=str, default='results.csv')  # will be TSV formatted
    args = parser.parse_args()

    seeds = [1234 + r * 7 for r in range(args.repeats)]

    # Your requested header (tab separated)
    header = [
        "ts","dataset","model",
        "acc-hat","acc","acc-val-hat","acc-val","acc-clean",
        "T-hat-RRE","T-hat",
        "acc-hat-std","acc-std","acc-val-hat-std","acc-val-std","acc-clean-std",
        "T-hat-RRE-std","T-hat-std"
    ]

    acc_list, val_acc_list, rre_list = [], [], []
    That_list = []  # store each run's T for std aggregation

    for seed in seeds:
        # mark runs/ before launching
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

        # read this run's result.json (authoritative)
        res = read_this_run_result(run_dir)
        test_acc     = res.get("test_acc", None)
        best_val_acc = res.get("best_val_acc", None)
        T_hat        = res.get("T", None)      # list of lists
        T_rre        = res.get("T_RRE", None)

        # accumulate stats
        if isinstance(test_acc, (int, float)):
            acc_list.append(float(test_acc))
        if isinstance(best_val_acc, (int, float)):
            val_acc_list.append(float(best_val_acc))
        if isinstance(T_rre, (int, float)):
            rre_list.append(float(T_rre))
        if isinstance(T_hat, list):
            That_list.append(T_hat)

        # write one row (acc-hat/acc、acc-val-hat/acc-val 用相同值；acc-clean暂无则留空)
        row = [
            ts_now, args.dataset, args.method,
            f"{test_acc:.4f}" if test_acc is not None else "",   # acc-hat
            f"{test_acc:.4f}" if test_acc is not None else "",   # acc
            f"{best_val_acc:.4f}" if best_val_acc is not None else "",  # acc-val-hat
            f"{best_val_acc:.4f}" if best_val_acc is not None else "",  # acc-val
            "",                                                   # acc-clean
            f"{T_rre:.6f}" if T_rre is not None else "",         # T-hat-RRE
            json.dumps(T_hat) if T_hat is not None else "",      # T-hat (json string)
            "", "", "", "", "", "", ""                           # std columns blank for per-run rows
        ]
        append_row_tsv(args.csv, header, row)
        print(f"[OK] seed={seed} appended.")

    # summary row with stds
    def mean_std(lst):
        if not lst:
            return 0.0, 0.0
        m = sum(lst)/len(lst)
        s = statistics.pstdev(lst) if len(lst) > 1 else 0.0
        return m, s

    acc_mean, acc_std = mean_std(acc_list)
    val_mean, val_std = mean_std(val_acc_list)
    rre_mean, rre_std = mean_std(rre_list)

    # T-hat-std（把每次 T 展平，计算 Frobenius 差的均值作为 std 的一个标量近似）
    That_std_str = ""
    if That_list:
        try:
            import numpy as np
            mats = [np.array(T) for T in That_list]
            shapes = {M.shape for M in mats}
            if len(shapes) == 1:
                A = np.stack(mats, axis=0)           # (R, C, C)
                mu = A.mean(axis=0, keepdims=True)   # (1, C, C)
                diffs = A - mu                       # (R, C, C)
                fro_each = np.linalg.norm(diffs.reshape(len(mats), -1), axis=1)  # (R,)
                That_std_str = f"{fro_each.mean():.6f}"  # 一个标量化 std 指标
        except Exception:
            That_std_str = ""

    ts_now = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_row = [
        ts_now, args.dataset, args.method,
        f"{acc_mean:.4f}", f"{acc_mean:.4f}",
        f"{val_mean:.4f}", f"{val_mean:.4f}",
        "",
        f"{rre_mean:.6f}" if rre_list else "",
        "",  # 不在汇总行放 T-hat 矩阵
        f"{acc_std:.4f}", f"{acc_std:.4f}",
        f"{val_std:.4f}" if val_acc_list else "",
        f"{val_std:.4f}" if val_acc_list else "",
        "",
        f"{rre_std:.6f}" if rre_list else "",
        That_std_str
    ]
    append_row_tsv(args.csv, header, summary_row)
    print(f"[OK] summary appended → {args.csv}")

if __name__ == "__main__":
    main()
