# Filter Construction Efficiency Report

This report analyzes the peak physical memory (Resident Set Size) required to build various range filters. 
Traditional metrics often only report the final filter size (Bits Per Key), but for production systems (like LSM-tree compactions), the **construction overhead** is a critical factor.

## Benchmark Configuration
- **Dataset**: `uniform` distribution
- **Key Count ($N$)**: $2^{24}$ (16,777,216 keys)
- **Range Length ($L$)**: 128
- **Target Accuracy**: $FPR \le 0.01$ (where possible)
- **Environment**: macOS (Apple M4 Max), measured using RSS Polling Monitor.

## Build Memory Overhead

| Algorithm | Parameter | FPR | Final BPK | **Peak RSS (Bytes/Key)** | Total Peak RSS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SODA-PEF** | $K=40$ | 0.0018 | 18.70 | **82.6** | 1.3 GB |
| **Scan-ARE-PEF** | $K=40$ | 0.0018 | 18.70 | **90.2** | 1.4 GB |
| **SODA (Basic)** | $K=40$ | 0.0018 | 21.58 | **184.2** | 2.9 GB |
| **Scan-ARE-SODA** | $K=40$ | 0.0018 | 21.58 | **184.3** | 2.9 GB |
| **BloomARE** | $\epsilon=0.01$ | 0.0100 | 9.57 | **~190.0** | 3.1 GB |
| **SNARF** | $BPK=18$ | **0.4999** | 17.38 | **123.1** | 1.9 GB |
| **Grafite (Fixed)** | $BPK=16$ | 0.0077 | 16.62 | **238.3** | 3.8 GB |
| **Rosetta** | $BPK=18$ | **0.0308** | 18.00 | **245.5** | 3.9 GB |
| **SuRF (Real 8)** | $bits=8$ | **0.5001** | 42.83 | **914.4** | 14.6 GB |

## Key Insights

1. **Efficiency of ARE-PEF**: The combination of SODA/ARE with the PEF (Elias-Fano) backend developed in this thesis is the most memory-efficient to construct. It requires only **82.6 bytes per key**, which is 3x less than industry alternatives like Grafite or Rosetta.
2. **Construction vs. Final Size**: While SuRF's final size is ~42 BPK, its construction requires **914 bytes per key** (~174x the final size). This "construction tax" makes it dangerous for high-throughput write-heavy systems.
3. **Robustness**: ARE-family filters maintain low construction memory while successfully achieving high accuracy ($FPR < 0.01$) on large ranges ($L=128$), whereas SNARF and SuRF fail to filter effectively at this range length.

---
*Generated on 2026-05-09*
