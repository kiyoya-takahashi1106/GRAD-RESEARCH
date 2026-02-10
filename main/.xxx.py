import re
import statistics
from pathlib import Path

model_type = "ablation"   # help: "clap", "method", "method2",  "ablation"
datasets = ["esc50", "us8k", "beijing_opera", "vocal_sound", "tut2017"]
seeds = [41, 42, 43]

def extract_acc(path: Path) -> float:
    text = path.read_text()
    m = re.search(r"Zero-shot Accuracy:\s*([0-9.]+)", text)
    if not m:
        raise ValueError(f"Accuracy not found in {path}")
    return float(m.group(1))

means = []
for dataset in datasets:
    log_dir = Path("logs") / "test_768" / f"{model_type}_mix"
    vals = [extract_acc(log_dir / f"{dataset}_{seed}.log") for seed in seeds]
    mean = statistics.mean(vals)
    stdev = statistics.pstdev(vals)  # population std
    means.append(mean)
    print(f"{dataset}: {mean:.4f}±{stdev:.4f}")

overall_mean = statistics.mean(means)
print(f"average_of_means: {overall_mean:.4f}")