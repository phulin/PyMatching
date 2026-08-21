#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: compare_differential.py BASE.csv LAZY.csv")

def load(path):
    rows = {}
    with Path(path).open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            case = int(row[0])
            rows[case] = (row[1], int(row[2]))
    return rows

base = load(sys.argv[1])
lazy = load(sys.argv[2])
if base.keys() != lazy.keys():
    raise SystemExit("case sets differ")
for case in sorted(base):
    if base[case] != lazy[case]:
        raise SystemExit(f"case {case}: base={base[case]} lazy={lazy[case]}")
print(f"matched_cases={len(base)}")
