#!/bin/python
# -*- coding: utf-8 -*-

import csv
import re

from common import ensure_dirs, model_dir


SUMMARY_FILE = "model_summary.csv"
BEST_FILE = "best_model.txt"

MODEL_RE = re.compile(
    r"^sunImputer_"
    r"(?P<epoch_begin>\d+)-(?P<epoch_end>\d+)epoch_"
    r"(?P<learning_rate>[-+0-9.eE]+)learningRate_"
    r"(?P<neurons>\d+)neurons_"
    r"(?P<l2>[-+0-9.eE]+)l2_"
    r"(?P<dropout>[-+0-9.eE]+)dropout_"
    r"(?P<batch_size>\d+)batchSize_"
    r"(?:(?P<train_num>[-+0-9.eE]+)train_(?P<val_num>[-+0-9.eE]+)val_)?"
    r"(?P<epoch>\d+)-(?P<val_loss>[-+0-9.eE]+)\.keras$"
)


def parse_model(path):
    match = MODEL_RE.match(path.name)
    if match is None:
        return None

    row = match.groupdict()
    row["model"] = path.name
    row["epoch_begin"] = int(row["epoch_begin"])
    row["epoch_end"] = int(row["epoch_end"])
    row["learning_rate"] = float(row["learning_rate"])
    row["neurons"] = int(row["neurons"])
    row["l2"] = float(row["l2"])
    row["dropout"] = float(row["dropout"])
    row["batch_size"] = int(row["batch_size"])
    row["train_num"] = 0.6 if row["train_num"] is None else float(row["train_num"])
    row["val_num"] = 0.2 if row["val_num"] is None else float(row["val_num"])
    row["epoch"] = int(row["epoch"])
    row["val_loss"] = float(row["val_loss"])
    return row


def main():
    ensure_dirs()
    out_dir = model_dir()
    rows = []

    for path in out_dir.glob("sunImputer_*.keras"):
        row = parse_model(path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No sunImputer_*.keras model found in {out_dir}")

    rows.sort(key=lambda row: (row["val_loss"], row["epoch"], row["model"]))
    best = rows[0]

    fields = [
        "model",
        "val_loss",
        "epoch",
        "epoch_begin",
        "epoch_end",
        "learning_rate",
        "neurons",
        "l2",
        "dropout",
        "batch_size",
        "train_num",
        "val_num",
    ]
    summary_path = out_dir / SUMMARY_FILE
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    best_path = out_dir / BEST_FILE
    best_path.write_text(best["model"] + "\n", encoding="utf-8")

    print("best sun imputer:", best["model"])
    print("val_loss:", best["val_loss"])
    print("summary:", summary_path)
    print("best file:", best_path)


if __name__ == "__main__":
    main()
