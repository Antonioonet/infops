import csv
import json
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="russia")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1] / "benchmark_results" / args.dataset
rows = []
for path in sorted(root.glob("*.json")):
    if path.name.endswith("_summary.json"):
        continue
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "auc" not in data:
        continue
    rows.append({
        "model": data.get("model", path.stem),
        "auc": data["auc"],
        "macro_f1": data["macro_f1"],
        "threshold": data.get("threshold", 0.5),
        "feature_dim": data.get("feature_dim"),
        "runtime_seconds": data.get("runtime_seconds"),
        "model_size_mb": data.get("model_size_mb"),
        "num_parameters": data.get("num_parameters"),
        "num_nodes": data.get("num_nodes", "split-test-mean"),
        "num_anomalies": data.get("num_anomalies", "split-test-mean"),
    })

rows.sort(key=lambda row: row["auc"], reverse=True)
out = root / "summary.csv"
with open(out, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(f"{row['model']:16s} auc={row['auc']:.6f} macro_f1={row['macro_f1']:.6f}")
print(f"Wrote {out}")
