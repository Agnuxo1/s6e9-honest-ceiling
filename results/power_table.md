| Planted structure | Groups | Effect d | Oracle AUC gain | True program on reserved fold | Found in 30 min |
|---|---|---|---|---|---|
| A: income mod 97 | 97 | 0.002 | 0.0002 | 0.0043 (fails 0.012) | no (below resolution) |
| A: income mod 97 | 97 | 0.004 | 0.0007 | 0.0100 (fails 0.012) | no (below resolution) |
| A: income mod 97 | 97 | 0.008 | 0.0019 | 0.0197 (passes 0.012) | **yes** (235 s) |
| A: income mod 97 | 97 | 0.016 | 0.0049 | 0.0427 (passes 0.012) | **yes** (35 s) |
| B: income hundreds digit × city | 30 | 0.004 | 0.0008 | 0.0079 (fails 0.012) | no (below resolution) |
| B: income hundreds digit × city | 30 | 0.008 | 0.0021 | 0.0187 (passes 0.012) | **yes** (34 s) |
| C: commute mod 37 × age mod 7 | 259 | 0.008 | 0.0021 | 0.0153 (passes 0.012) | **no** (search miss) |
| C: commute mod 37 × age mod 7 | 259 | 0.016 | 0.0048 | 0.0390 (passes 0.012) | **yes** (118 s) |
