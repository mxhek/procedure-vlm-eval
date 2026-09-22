# scores model predictions against the labels

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = PROJECT_ROOT / "labels" / "labels.csv"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_CSV = PROJECT_ROOT / "results" / "metrics.csv"

STEPS = ["take_cup", "add_teabag", "pour_water"]
INVALID = "invalid"  # any missing, unparseable or unknown answer


def load_labels(split):
    df = pd.read_csv(LABELS_CSV)
    df["verified"] = df["verified"].fillna("").astype(str).str.strip().str.lower()
    if split != "all":
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


def normalise_step(value):
    # Turns a model answer into one of STEPS, or INVALID.
    if not isinstance(value, str):
        return INVALID
    step = value.strip().lower().replace(" ", "_").replace("-", "_")
    return step if step in STEPS else INVALID


def load_predictions(experiment):
    path = RAW_DIR / f"{experiment}.jsonl"
    if not path.exists():
        raise SystemExit(f"No predictions found at {path}")

    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    preds = pd.DataFrame(rows)
    # If a frame was run more than once, keep the latest answer
    preds = preds.drop_duplicates(subset="frame", keep="last")
    preds["pred"] = preds["step"].map(normalise_step)
    if "confidence" not in preds:
        preds["confidence"] = None
    return preds[["frame", "pred", "confidence"]]


def majority_baseline(labels):
    most_common = labels["step"].value_counts().idxmax()
    out = labels.copy()
    out["pred"] = most_common
    out["confidence"] = None
    return out, most_common


# metrics
def wilson_interval(correct, n, z=1.96):
    """95% confidence interval for accuracy. Wide intervals = small sample."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def accuracy(df):
    return (df["pred"] == df["step"]).mean() if len(df) else float("nan")


def compute_metrics(df):
    df = df.assign(correct=df["pred"] == df["step"])
    n, n_correct = len(df), int(df["correct"].sum())
    low, high = wilson_interval(n_correct, n)

    per_step = {step: df.loc[df["step"] == step, "correct"].mean() for step in STEPS}
    present = [v for v in per_step.values() if not math.isnan(v)]

    clear = df[df["verified"] == "yes"]
    ambiguous = df[df["verified"].isin(["no", "unclear"])]

    conf = pd.to_numeric(df["confidence"], errors="coerce")

    return {
        "n": n,
        "accuracy": n_correct / n,
        "ci_low": low,
        "ci_high": high,
        "balanced_accuracy": sum(present) / len(present),
        "invalid_rate": (df["pred"] == INVALID).mean(),
        "acc_clear_frames": accuracy(clear),
        "n_clear": len(clear),
        "acc_ambiguous_frames": accuracy(ambiguous),
        "n_ambiguous": len(ambiguous),
        **{f"recall_{s}": v for s, v in per_step.items()},
        "mean_conf_correct": conf[df["correct"]].mean(),
        "mean_conf_wrong": conf[~df["correct"]].mean(),
    }, df


# output
def save_confusion_matrix(df, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = STEPS + ([INVALID] if (df["pred"] == INVALID).any() else [])
    matrix = pd.crosstab(df["step"], df["pred"]).reindex(index=STEPS, columns=cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 2, 4.5))
    ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(STEPS)), STEPS)
    ax.set_xlabel("Model prediction")
    ax.set_ylabel("True step")
    ax.set_title(name)
    top = matrix.values.max()
    for i in range(len(STEPS)):
        for j in range(len(cols)):
            v = matrix.values[i, j]
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > top / 2 else "black")
    fig.tight_layout()
    path = FIG_DIR / f"{name}_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return matrix, path


def append_metrics(row):
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV)
        # Re-scoring the same experiment on the same split replaces the old row
        old = old[~((old["experiment"] == row["experiment"]) & (old["split"] == row["split"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.round(4).to_csv(METRICS_CSV, index=False)


def pct(x):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:5.1%}"


def print_report(name, m, matrix, fig_path, note=""):
    print(f"\n=== {name} ===")
    if note:
        print(note)
    print(f"Frames:             {m['n']}")
    print(f"Accuracy:           {pct(m['accuracy'])}   (95% CI {pct(m['ci_low'])} to {pct(m['ci_high'])})")
    print(f"Balanced accuracy:  {pct(m['balanced_accuracy'])}")
    print(f"Invalid outputs:    {pct(m['invalid_rate'])}")
    print(f"Clear frames:       {pct(m['acc_clear_frames'])}   (n={m['n_clear']})")
    print(f"Ambiguous frames:   {pct(m['acc_ambiguous_frames'])}   (n={m['n_ambiguous']})")
    print("Per-step accuracy:")
    for s in STEPS:
        print(f"  {s:<12} {pct(m['recall_' + s])}")
    if not math.isnan(m["mean_conf_correct"]):
        print(f"Mean confidence: correct {m['mean_conf_correct']:.2f}, wrong {m['mean_conf_wrong']:.2f}")
    print("\nConfusion matrix (rows = true step, columns = prediction):")
    print(matrix.to_string())
    print(f"\nSaved figure: {fig_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Score model predictions against labels.")
    parser.add_argument("experiment", nargs="?", help="experiment name (file in results/raw/)")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--baseline", action="store_true", help="score the majority-class baseline")
    args = parser.parse_args()

    labels = load_labels(args.split)

    if args.baseline:
        df, guess = majority_baseline(labels)
        experiment, note = "baseline_majority", f"Always predicts: {guess}"
    elif args.experiment:
        preds = load_predictions(args.experiment)
        df = labels.merge(preds, on="frame", how="left")
        missing = df["pred"].isna().sum()
        df["pred"] = df["pred"].fillna(INVALID)  # no answer counts as wrong
        experiment = args.experiment
        note = f"WARNING: {missing} frames had no prediction (counted as invalid)" if missing else ""
    else:
        parser.error("give an experiment name or --baseline")

    name = f"{experiment}_{args.split}"
    metrics, df = compute_metrics(df)
    matrix, fig_path = save_confusion_matrix(df, name)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df[["frame", "participant", "step", "pred", "correct", "confidence", "verified"]] \
        .to_csv(PRED_DIR / f"{name}.csv", index=False)

    append_metrics({"experiment": experiment, "split": args.split, **metrics})
    print_report(name, metrics, matrix, fig_path, note)


if __name__ == "__main__":
    main()