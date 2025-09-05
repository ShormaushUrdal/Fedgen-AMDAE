import argparse
import glob
import json
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Candidate keys to discover labels/predictions in your HDF5 files
CANDIDATE_TRUE_KEYS = ["y_true", "test_y", "labels", "targets", "test_targets"]
CANDIDATE_PRED_KEYS = ["y_pred", "preds", "test_pred", "test_predictions", "predictions"]
CANDIDATE_PROBA_KEYS = ["y_prob", "probs", "probabilities", "logits", "outputs"]

# HDF5 magic signature (first 8 bytes)
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


# ---------------------------
# Utility helpers
# ---------------------------
def is_hdf5(path: str) -> bool:
    """Quick signature check to avoid h5py OSError on non-HDF5 files."""
    try:
        with open(path, "rb") as f:
            return f.read(8) == HDF5_MAGIC
    except OSError:
        return False


def _first_existing_key(hf: h5py.File, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in hf:
            return k
    return None


def _ensure_1d_labels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.reshape(-1)
    return x


def _proba_to_pred(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probabilities/logits to class predictions.
    Binary: shape (N,) or (N,1) -> thresholding.
    Multiclass: shape (N,C) -> argmax along C.
    """
    proba = np.asarray(proba)
    if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
        return (proba.reshape(-1) >= threshold).astype(int)
    elif proba.ndim == 2:
        return np.argmax(proba, axis=1)
    else:
        # Fallback for higher-rank arrays: take argmax on last axis and flatten
        pred = np.argmax(proba, axis=-1)
        return pred.reshape(-1)


def load_labels_and_preds(h5_path: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Open a valid HDF5 file, find y_true and y_pred (or probs->pred), and return them
    along with discovered labels and the list of file keys for debugging.
    """
    if not is_hdf5(h5_path):
        raise OSError("Not an HDF5 file (magic signature mismatch)")

    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())

        true_key = _first_existing_key(hf, CANDIDATE_TRUE_KEYS)
        pred_key = _first_existing_key(hf, CANDIDATE_PRED_KEYS)
        proba_key = _first_existing_key(hf, CANDIDATE_PROBA_KEYS)

        if true_key is None:
            raise KeyError(f"Missing ground-truth; keys present: {keys}")

        y_true = _ensure_1d_labels(hf[true_key][()])

        if pred_key is not None:
            y_pred = _ensure_1d_labels(hf[pred_key][()])
        elif proba_key is not None:
            y_pred = _proba_to_pred(hf[proba_key][()])
            y_pred = _ensure_1d_labels(y_pred)
        else:
            raise KeyError(f"Missing predictions/probabilities; keys present: {keys}")

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Length mismatch y_true={y_true.shape} vs y_pred={y_pred.shape}")

        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    return y_true, y_pred, labels, keys


def save_confusion_and_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    out_stub: str,
    normalize: str = "none",
    save_fig: bool = True,
):
    """
    Compute confusion matrix and F1 aggregates, then save CSV/JSON and optional figure.
    """
    norm_arg = None if normalize == "none" else normalize
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm_arg)

    # Save confusion matrix as CSV
    import csv
    cm_csv = out_stub + "_confusion_matrix.csv"
    with open(cm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"pred_{l}" for l in labels])
        for i, row in enumerate(cm):
            writer.writerow([f"true_{labels[i]}"] + list(row))

    # Save classification report
    from sklearn.metrics import classification_report
    import pandas as pd
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(out_stub + "_classification_report.csv", index=True)

    # F1 aggregates
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    summary = {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "accuracy_report": report.get("accuracy", np.nan),
    }
    with open(out_stub + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Optional PNG
    if save_fig:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", colorbar=True, values_format=".2f" if norm_arg else "d")
        plt.tight_layout()
        plt.savefig(out_stub + "_confusion_matrix.png", dpi=300)
        plt.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute F1 and confusion matrices from HDF5 result files.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing .h5 files.")
    parser.add_argument("--glob", type=str, default="*.h5", help="Glob pattern, e.g., \"*.h5\".")
    parser.add_argument("--output_dir", type=str, default="eval", help="Subdirectory under results_dir for outputs.")
    parser.add_argument("--normalize", type=str, choices=["none", "true", "pred", "all"], default="none",
                        help="Normalization for confusion_matrix.")
    parser.add_argument("--save_figs", type=int, default=1, help="Save PNG confusion matrices (1=yes,0=no).")
    args = parser.parse_args()

    # Discover files
    pattern = os.path.join(args.results_dir, args.glob)
    h5_paths = sorted(glob.glob(pattern))
    if not h5_paths:
        print(f"No .h5 files found under {args.results_dir} with pattern {args.glob}")
        return

    out_root = os.path.join(args.results_dir, args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    # Process each file
    all_rows = []
    for p in h5_paths:
        base, _ = os.path.splitext(os.path.basename(p))
        out_stub = os.path.join(out_root, base)
        try:
            y_true, y_pred, labels, keys = load_labels_and_preds(p)
            summary = save_confusion_and_f1(
                y_true, y_pred, labels, out_stub, normalize=args.normalize, save_fig=bool(args.save_figs)
            )
            all_rows.append({"file": base, **summary})
            print(f"[OK] {base} (labels={len(labels)}) -> {out_stub}_*.csv/png/json | keys={keys}")
        except (OSError, KeyError, ValueError) as e:
            print(f"[SKIP] {base}: {e}")

    # Summary CSV across files
    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(out_root, "summary_f1_confusion.csv"), index=False)
        print(f"Wrote summary to {os.path.join(out_root, 'summary_f1_confusion.csv')}")


if __name__ == "__main__":
    main()
