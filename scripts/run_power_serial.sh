#!/bin/bash
# Remaining power runs, one at a time (shared machine), after the two running ones finish.
D=${DATA_DIR:?set DATA_DIR to the folder with train.csv, test.csv and the original table}
B=${BLEND_OOF:?set BLEND_OOF to the nested blend OOF array (.npy)}
while [ "$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*power_analysis*' }).Count")" != "0" ]; do sleep 60; done
for spec in "A 0.008" "B 0.004" "C 0.008"; do
  set -- $spec
  python scripts/power_analysis.py --data-dir $D --blend-oof $B --plant $1 --effect $2 --minutes 30 --out results/power/power_$1_$2.json > logs/power_$1_$2.log 2>&1
done
