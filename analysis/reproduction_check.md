# Reproduction check (run 1 vs run 2)

Same seed base, same per-trial seeds. Tolerance for per-trial values: 0.005 (GPU float32). Reported means must agree at three decimals.

| File | Max abs per-trial difference | Mismatched rounded means | Verdict |
|---|---|---|---|
| multiseed_label2.json | 3.16e-06 | 0 | PASS |
| multiseed_label1.json | 7.65e-05 | 0 | PASS |
| sage_label2.json | 2.21e-03 | 0 | PASS |
| sage_label1.json | 4.05e-07 | 0 | PASS |

Overall: **PASS**
