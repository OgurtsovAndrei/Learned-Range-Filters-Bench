# ARE Exact Backend Comparison

Filters: SODA, Greedy+Merge

n=1048576, rangeLen=4096, epsilon=0.0100, mixed query workload (32768 queries, 3 rounds)

| Filter | Dataset | n | Classic build ms | One-D build ms | Classic query ns | One-D query ns | Speedup | Classic bpk | One-D bpk | Delta bpk |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SODA | uniform | 1048576 | 256.79 | 258.61 | 164.00 | 133.57 | 1.23x | 22.000 | 21.000 | 1.000 |
| Greedy+Merge | uniform | 1048576 | 84.81 | 91.07 | 222.60 | 159.53 | 1.40x | 22.000 | 21.000 | 1.000 |
| SODA | clustered | 1048576 | 174.77 | 176.75 | 331.33 | 322.27 | 1.03x | 22.000 | 21.000 | 1.000 |
| Greedy+Merge | clustered | 1048576 | 63.92 | 55.72 | 474.78 | 344.38 | 1.38x | 17.696 | 16.961 | 0.734 |
| SODA | sosd_fb | 1048576 | 37.95 | 35.85 | 671.93 | 566.48 | 1.19x | 22.000 | 21.000 | 1.000 |
| Greedy+Merge | sosd_fb | 1048576 | 56.46 | 56.49 | 250.56 | 331.06 | 0.76x | 12.000 | 11.000 | 1.000 |
| SODA | sosd_wiki | 988851 | 32.12 | 31.99 | 630.27 | 586.65 | 1.07x | 22.060 | 21.530 | 0.530 |
| Greedy+Merge | sosd_wiki | 988851 | 51.86 | 57.75 | 299.52 | 359.13 | 0.83x | 10.061 | 9.530 | 0.530 |
| SODA | sosd_osm | 1048576 | 212.76 | 221.79 | 365.04 | 196.08 | 1.86x | 22.000 | 21.500 | 0.500 |
| Greedy+Merge | sosd_osm | 1048576 | 113.58 | 111.45 | 450.82 | 418.10 | 1.08x | 33.263 | 32.525 | 0.738 |
| SODA | sosd_books | 1048576 | 49.21 | 36.75 | 590.96 | 584.60 | 1.01x | 22.000 | 21.000 | 1.000 |
| Greedy+Merge | sosd_books | 1048576 | 60.34 | 65.84 | 283.79 | 189.92 | 1.49x | 5.000 | 4.000 | 1.000 |

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
