#!/bin/python
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path


MODEL_DIR = Path('./Data/modelw')
SUMMARY_FILE = MODEL_DIR / 'model_summary.csv'
BEST_FILE = MODEL_DIR / 'best_model.txt'

MODEL_RE = re.compile(
    r'^errWeighted_'
    r'(?P<epoch_begin>\d+)-(?P<epoch_end>\d+)epoch_'
    r'(?P<learning_rate>[-+0-9.eE]+)learningRate_'
    r'(?P<neurons>\d+)neurons_'
    r'(?P<l2>[-+0-9.eE]+)l2_'
    r'(?P<dropout>[-+0-9.eE]+)dropout_'
    r'(?P<batch_size>\d+)batchSize_'
    r'(?P<epoch>\d+)-(?P<val_loss>[-+0-9.eE]+)\.keras$'
)


def parse_model(path):
    match = MODEL_RE.match(path.name)
    if not match:
        return None

    row = match.groupdict()
    row['model'] = path.name
    row['epoch_begin'] = int(row['epoch_begin'])
    row['epoch_end'] = int(row['epoch_end'])
    row['learning_rate'] = float(row['learning_rate'])
    row['neurons'] = int(row['neurons'])
    row['l2'] = float(row['l2'])
    row['dropout'] = float(row['dropout'])
    row['batch_size'] = int(row['batch_size'])
    row['epoch'] = int(row['epoch'])
    row['val_loss'] = float(row['val_loss'])
    return row


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in MODEL_DIR.glob('errWeighted_*.keras'):
        row = parse_model(path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f'No weighted keras model found in {MODEL_DIR}')

    rows.sort(key=lambda row: (row['val_loss'], row['epoch'], row['model']))
    best = rows[0]

    fields = [
        'model', 'val_loss', 'epoch', 'epoch_begin', 'epoch_end',
        'learning_rate', 'neurons', 'l2', 'dropout', 'batch_size',
    ]
    with SUMMARY_FILE.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    BEST_FILE.write_text(best['model'] + '\n', encoding='utf-8')

    print('best weighted model:', best['model'])
    print('val_loss:', best['val_loss'])
    print('summary:', SUMMARY_FILE)
    print('best file:', BEST_FILE)


if __name__ == '__main__':
    main()
