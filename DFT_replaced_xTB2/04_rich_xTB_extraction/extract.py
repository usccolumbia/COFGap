#!/usr/bin/env python3
"""Parallel, resumable rich-xTB extraction over the 591-COF dataset."""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "dataset_code"
DEFAULT_CIFS = DATA / "CoRE-COFs_DT591-v5.0"
DEFAULT_SCREENER = DATA / "pv_ranked_H.csv"
DEFAULT_PACKAGE = DATA
DEFAULT_FRAGMENTS = HERE.parents[1] / "DFT_cof_bandgap" / "fragments"


def run_one(key, source, package_dir, timeout):
    if not source.is_file():
        return key, {"error": f"missing source: {source.name}"}
    try:
        proc = subprocess.run([sys.executable, str(HERE / "worker.py"), str(source),
                               str(package_dir)], capture_output=True, text=True,
                              timeout=timeout)
        lines = [x for x in proc.stdout.splitlines() if x.strip().startswith("{")]
        return key, json.loads(lines[-1]) if lines else {
            "error": (proc.stderr or "worker produced no JSON")[-500:]}
    except subprocess.TimeoutExpired:
        return key, {"error": "timeout"}
    except Exception as exc:
        return key, {"error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIFS)
    parser.add_argument("--fragments-dir", type=Path, default=DEFAULT_FRAGMENTS)
    parser.add_argument("--geometry", choices=["fragments", "cif"], default="fragments")
    parser.add_argument("--targets", nargs="+", choices=["node", "linker", "nodelinker"],
                        default=["node", "linker", "nodelinker"])
    parser.add_argument("--screener", type=Path, default=DEFAULT_SCREENER)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--cache", type=Path, default=HERE / "rich_xtb_cache.json")
    parser.add_argument("--out", type=Path, default=HERE / "rich_xtb_features.csv")
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2)//2)))
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="development/testing limit; omit for all rows")
    parser.add_argument("--labeled-only", action="store_true",
                        help="compute only the 315 COFs with supervised DFT targets")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.screener.open(encoding="utf-8-sig")))
    if args.labeled_only:
        gap_lines = (DATA / "bandgap_CoRE_COF_v5.0_PCAdata.csv").read_text(
            encoding="utf-8-sig").splitlines()
        labeled = {line.split(";", 1)[0].strip() for line in gap_lines[1:] if line.strip()}
        rows = [row for row in rows if row["file"].replace(".cif", "") in labeled]
    if args.limit:
        rows = rows[:args.limit]
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    tasks = []
    for row in rows:
        cof_id = row["file"].replace(".cif", "")
        if args.geometry == "fragments":
            for target in args.targets:
                key = f"{row['file']}:{target}"
                tasks.append((key, args.fragments_dir / cof_id / f"{target}.xyz"))
        else:
            tasks.append((row["file"], args.cif_dir / row["file"]))
    todo = [(key, source) for key, source in tasks
            if key not in cache or (args.retry_errors and "error" in cache[key])]
    print(f"rows={len(rows)} tasks={len(tasks)} cached={len(tasks)-len(todo)} "
          f"to_compute={len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, key, source, args.package_dir,
                               args.timeout): key for key, source in todo}
        for i, future in enumerate(as_completed(futures), 1):
            key, value = future.result(); cache[key] = value
            if i % 10 == 0 or i == len(todo):
                args.cache.write_text(json.dumps(cache, indent=2, allow_nan=True))
                ok = sum("error" not in cache[k] for k, _ in tasks if k in cache)
                print(f"completed {i}/{len(todo)}; successful tasks={ok}", flush=True)
    args.cache.write_text(json.dumps(cache, indent=2, allow_nan=True))
    feature_names = sorted({name for value in cache.values()
                            for name in value if name != "error"})
    prefixes = args.targets if args.geometry == "fragments" else ["framework"]
    output_features = [f"{prefix}_{name}" for prefix in prefixes for name in feature_names]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "xtb_status", *output_features])
        writer.writeheader()
        for row in rows:
            results = {}
            statuses = []
            for prefix in prefixes:
                key = (f"{row['file']}:{prefix}" if args.geometry == "fragments"
                       else row["file"])
                result = cache.get(key, {"error": "not computed"})
                statuses.append(f"{prefix}:{result.get('error', 'OK')}")
                results.update({f"{prefix}_{k}": result.get(k, "")
                                for k in feature_names})
            status = "OK" if all(x.endswith(":OK") for x in statuses) else ";".join(statuses)
            writer.writerow({"file": row["file"], "xtb_status": status, **results})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
