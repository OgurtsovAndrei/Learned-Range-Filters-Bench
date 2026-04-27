# ARE Exact Backend Comparison

Filters: SODA, Greedy+Merge

n=1048576, rangeLen=4096, epsilon=0.0100, mixed query workload (32768 queries, 3 rounds)

| Filter | Dataset | n | Classic build ms | One-D build ms | Classic query ns | One-D query ns | Speedup | Classic bpk | One-D bpk | Delta bpk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SODA | uniform | 1048576 | 252.07 | 254.47 | 156.24 | 140.47 | 1.11x | 21.632 | 21.000 | 0.632 |
| Greedy+Merge | uniform | 1048576 | 102.91 | 91.73 | 199.72 | 169.96 | 1.18x | 21.632 | 21.000 | 0.632 |
| SODA | clustered | 1048576 | 191.94 | 186.50 | 370.00 | 314.24 | 1.18x | 21.110 | 21.000 | 0.110 |
| Greedy+Merge | clustered | 1048576 | 65.56 | 62.24 | 523.28 | 389.84 | 1.34x | 17.201 | 16.961 | 0.240 |
| SODA | sosd_fb | 1048576 | 41.03 | 40.71 | 703.01 | 618.96 | 1.14x | 21.001 | 21.000 | 0.001 |
| Greedy+Merge | sosd_fb | 1048576 | 70.49 | 60.95 | 251.06 | 322.66 | 0.78x | 11.232 | 11.000 | 0.232 |
| SODA | sosd_wiki | 988851 | 33.18 | 33.69 | 562.60 | 533.66 | 1.05x | 21.530 | 21.530 | 0.000 |
| Greedy+Merge | sosd_wiki | 988851 | 64.10 | 50.30 | 276.41 | 346.92 | 0.80x | 9.708 | 9.530 | 0.177 |
| SODA | sosd_osm | 1048576 | 199.25 | 220.08 | 306.16 | 162.44 | 1.88x | 21.930 | 21.500 | 0.430 |
| Greedy+Merge | sosd_osm | 1048576 | 119.76 | 115.91 | 453.06 | 427.85 | 1.06x | 32.817 | 32.525 | 0.293 |
| SODA | sosd_books | 1048576 | 56.84 | 39.83 | 656.68 | 635.21 | 1.03x | 21.000 | 21.000 | 0.000 |
| Greedy+Merge | sosd_books | 1048576 | 67.22 | 70.36 | 315.90 | 208.96 | 1.51x | 4.517 | 4.000 | 0.517 |

## Notes

- SODA / uniform: classic `K=39`, one_d `K=39`
- Greedy+Merge / uniform: classic `K=39 clusters=0 fallback=1048576`, one_d `K=39 clusters=0 fallback=1048576`
- SODA / clustered: classic `K=39`, one_d `K=39`
- Greedy+Merge / clustered: classic `K=39 clusters=8 fallback=104857`, one_d `K=39 clusters=8 fallback=104857`
- SODA / sosd_fb: classic `K=39`, one_d `K=39`
- Greedy+Merge / sosd_fb: classic `K=39 clusters=1 fallback=0`, one_d `K=39 clusters=1 fallback=0`
- SODA / sosd_wiki: classic `K=39`, one_d `K=39`
- Greedy+Merge / sosd_wiki: classic `K=39 clusters=1 fallback=0`, one_d `K=39 clusters=1 fallback=0`
- SODA / sosd_osm: classic `K=39`, one_d `K=39`
- Greedy+Merge / sosd_osm: classic `K=39 clusters=5335 fallback=34979`, one_d `K=39 clusters=5335 fallback=34979`
- SODA / sosd_books: classic `K=39`, one_d `K=39`
- Greedy+Merge / sosd_books: classic `K=39 clusters=1 fallback=0`, one_d `K=39 clusters=1 fallback=0`
