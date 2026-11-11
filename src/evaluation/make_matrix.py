#!/usr/bin/env python3
"""Build the cross-dataset RTD matrix from runs/*/results.json.

RTD(A->B) = (IoU(A,A) - IoU(A,B)) / IoU(A,A) * 100; diagonal = 0.
Multiple runs (seeds) for one training dataset -> mean +- std. Never crashes on partial runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ORDER = ("whu", "inria", "mass")


def collect(runs_dir: Path, arch: str) -> dict[str, dict[str, list[float]]]:
    """runs[train_set] -> test_set -> [iou, iou, ...] (over seeds)."""
    runs: dict[str, dict[str, list[float]]] = {t: {s: [] for s in ORDER} for t in ORDER}
    if not runs_dir.is_dir():
        print(f"WARNING: runs dir not found: {runs_dir}")
        return runs
    for res in sorted(runs_dir.glob("*/results.json")):
        try:
            data = json.loads(res.read_text())
        except json.JSONDecodeError as e:
            print(f"WARNING: unreadable {res}: {e}")
            continue
        if data.get("arch") != arch:
            continue
        train_set = data.get("train_set")
        if train_set not in ORDER:
            continue
        for test_set, metrics in (data.get("test") or {}).items():
            if test_set in ORDER and isinstance(metrics, dict) and "iou" in metrics:
                runs[train_set][test_set].append(float(metrics["iou"]))
    return runs


def fmt_cell(vals: list[float], diagonal: bool) -> str:
    if not vals:
        return "-"
    if diagonal:
        return f"{statistics.mean(vals):.4f}"
    m = statistics.mean(vals)
    if len(vals) > 1:
        s = statistics.stdev(vals)
        return f"{m:.4f}±{s:.4f}"
    return f"{m:.4f}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Cross-dataset IoU / RTD matrix")
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--arch", default="unet")
    p.add_argument("--out", default=None, help="output dir (default: runs_dir)")
    args = p.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out) if args.out else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect(runs_dir, args.arch)

    # ---- IoU matrix ----
    iou_rows = []
    for t in ORDER:
        iou_rows.append([fmt_cell(runs[t][s], diagonal=(t == s)) for s in ORDER])
    # RTD matrix (use the mean of each cell)
    rtd_rows = []
    for t in ORDER:
        diag_vals = runs[t][t]
        diag = statistics.mean(diag_vals) if diag_vals else None
        row = []
        for s in ORDER:
            vals = runs[t][s]
            if not vals:
                row.append("-")
            elif t == s or diag in (None, 0.0):
                row.append("0.0000" if t == s else "-")
            else:
                row.append(f"{(diag - statistics.mean(vals)) / diag * 100:.2f}")
        rtd_rows.append(row)

    def md_table(header, rows):
        lines = ["| " + " | ".join(["train\\test", *header]) + " |",
                 "|" + "|".join(["---"] * (len(header) + 1)) + "|"]
        for name, row in zip(header, rows):
            lines.append("| " + " | ".join([name, *row]) + " |")
        return "\n".join(lines)

    print(f"\n== IoU matrix (arch={args.arch}) ==")
    print(md_table(ORDER, iou_rows), "\n")
    print("== RTD % matrix ==")
    print(md_table(ORDER, rtd_rows))

    (out_dir / "matrix_iou.csv").write_text(
        "train_test," + ",".join(ORDER) + "\n" +
        "\n".join(f"{t}," + ",".join(r) for t, r in zip(ORDER, iou_rows)) + "\n")
    (out_dir / "matrix_rtd.csv").write_text(
        "train_test," + ",".join(ORDER) + "\n" +
        "\n".join(f"{t}," + ",".join(r) for t, r in zip(ORDER, rtd_rows)) + "\n")
    (out_dir / "matrix.md").write_text(
        f"# Cross-dataset matrices (arch={args.arch})\n\n## IoU\n\n"
        + md_table(ORDER, iou_rows) + "\n\n## RTD %\n\n" + md_table(ORDER, rtd_rows) + "\n")
    print(f"\nwritten: matrix_iou.csv, matrix_rtd.csv, matrix.md in {out_dir}")


if __name__ == "__main__":
    main()
