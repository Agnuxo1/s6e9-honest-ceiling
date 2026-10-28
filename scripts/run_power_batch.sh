#!/bin/bash
# Re-run power analysis in groups of three to stay within RAM.
D=${DATA_DIR:?set DATA_DIR to the folder with train.csv, test.csv and the original table}
B=${BLEND_OOF:?set BLEND_OOF to the nested blend OOF array (.npy)}
run() { python scripts/power_analysis.py --data-dir $D --blend-oof $B --plant $1 --effect $2 --minutes 30 --out results/power/power_$1_$2.json > logs/power_$1_$2.log 2>&1; }
run A 0.002 & run A 0.004 & run A 0.008 & wait
run B 0.004 & run C 0.008 & run B 0.002 & wait
