# FallbackInGapFPR — fallback-policy selection for SegARE

**Date:** 2026-05-15
**Branch:** `worktree-fallback-ingap-fpr`
**Status:** Design approved, awaiting writing-plans

## 1. Goal

Replace the unconditional `FallbackAlwaysSODA{}` policy used in `SegARE` with a
**data-aware** policy that selects between `TruncARE` and `AdaptiveARE` (SODA) for
the sparse fallback key set, based on the **proven** per-gap FPR formula from
the thesis (Section 5.2, Eq. labelled `eq:trunc-fpr`).

The selection must:

1. Give an explicit guarantee `E[FPR] ≤ ε` under the benchmark `in-gap` query
   distribution (the distribution used in our SOSD comparison test suite).
2. Run in `O(n)` over the already-sorted fallback key set.
3. Plug in via the existing `hybridutil.FallbackPolicy` interface with **no
   interface changes**.

## 2. Background

### 2.1 Per-gap FPR formula (thesis Section 5.2)

For a single gap `R = keys[i] - keys[i-1]` between two consecutive sorted keys,
with the query placed uniformly inside that gap and `L = rangeLen`, the FPR of
the TruncARE filter is

```
FPR(R) = 0                                  if R ≤ L
FPR(R) = min(1, 2^t / (R - L))              if R > L
```

where `2^t = spread >> K = P` is the size of the phantom block (the universe
slice that hashes to a single fingerprint).

### 2.2 The benchmark query model

`bench/internal/querygen/querygen.go:208-222` (`GenerateSmartQueriesWeighted`,
in-gap bucket):

```go
g := gaps[rng.Intn(len(gaps))]              // gap chosen UNIFORMLY BY INDEX
a := g.lo + randUint64Below(rng, gapLen)    // query uniform inside the gap
```

Conditional on this generator, the expected FPR over the in-gap query bucket is

```
E[FPR]_in-gap = (1/(n-1)) · Σ_i FPR(R_i)
              = (1/(n-1)) · Σ_i min(1, P/(R_i - L))            (Model A)
```

This is the **mean-aggregation** model. The user has explicitly chosen `Model A`
over the worst-case alternative (`min R_i > L + P/ε`) — see brainstorming session.

### 2.3 Current state of the codebase

Five fallback policies exist in `Thesis/emptiness/approx/hybrid/hybridutil/policy.go`:

| Policy | Predicate | Problem |
|---|---|---|
| `FallbackEstimateFPR{ε}` | `n/2^K ≤ ε` | Uniform-over-universe model; unsafe for in-gap. |
| `FallbackGapFraction{ε}` | `Σ_{g≤P} g / spread ≤ ε` | Heuristic, no proof correspondence. |
| `FallbackPhantom` | `P < L` | Necessary, never sufficient. |
| `FallbackAuto` (`TruncSafe`) | `P5(gaps) > P` | 5%-quantile baked in — safe only for ε ≥ 5%. |
| `FallbackAlwaysSODA` / `FallbackAlwaysTrunc` | constant | trivial baselines |

None directly evaluates the in-gap FPR formula.

`seg_are.go:86` currently hardcodes `FallbackAlwaysSODA{}` — always pays the
`log₂(L) ≈ 7` BPK SODA overhead, even on uniform / sparse fallback sets where
Trunc would be safe.

### 2.4 Why `t` inside AdaptiveARE is not the answer

(Bonus question from the brainstorming session.) Internal truncation `t > 0`
inside AdaptiveARE before its SODA hash either:

- does nothing for sparse fallback sets (`R_i ≫ 2^t` ⇒ `n' = n` ⇒ SODA FPR
  unchanged); or
- makes AdaptiveARE collapse to TruncARE-via-exact-mode (when `t = s − K`),
  which is the case we have just rejected by choosing SODA.

So when SODA is selected as fallback, the optimal internal `t* = 0`. The
current code already does this. **Out of scope for this design.**

## 3. Design

### 3.1 New policy type

In `Thesis/emptiness/approx/hybrid/hybridutil/policy.go`:

```go
// FallbackInGapFPR selects Trunc iff the expected in-gap FPR of TruncARE
// on the given fallback key set does not exceed Epsilon.
//
// The predicate evaluates, in one O(n) pass over the sorted keys,
//
//     (1/(n-1)) · Σ_i min(1, P / max(1, R_i - L))   ≤  Epsilon
//
// where R_i is the i-th consecutive gap and P = spread >> K is the
// truncation phantom-block size. This matches the per-gap FPR of
// Eq. 5.2 of the thesis and the in-gap query distribution used by
// bench/internal/querygen.GenerateSmartQueriesWeighted (gap chosen
// uniformly by index, query placed uniformly inside the gap).
//
// Guarantee (Expected FPR, Model A): if useTrunc returns true,
// then E[FPR_trunc]_in-gap ≤ Epsilon + O(1/2^t).
type FallbackInGapFPR struct{ Epsilon float64 }

func (FallbackInGapFPR) String() string { return "InGapFPR" }
```

### 3.2 Algorithm

```go
func (f FallbackInGapFPR) useTrunc(keys []uint64, K uint32, L uint64) bool {
    n := len(keys)
    if n < 2 {
        return true
    }
    spread := keys[n-1] - keys[0]
    if spread == 0 {
        return true
    }
    // Exact-mode path inside TruncARE → FPR = 0 by construction.
    spreadBits := uint32(64 - mbits.LeadingZeros64(spread))
    if spreadBits <= K {
        return true
    }
    P := spread >> K          // phantom block ≈ 2^t
    if P == 0 {
        P = 1
    }

    var sum float64
    for i := 1; i < n; i++ {
        R := keys[i] - keys[i-1]
        switch {
        case R <= L:
            // FPR_i = 0  (no empty query of length L fits in this gap)
        case R-L <= P:
            sum += 1.0                          // saturated: 2^t/(R-L) ≥ 1
        default:
            sum += float64(P) / float64(R-L)    // Eq. 5.2
        }
    }
    return sum/float64(n-1) <= f.Epsilon
}
```

Three branches reflect the three regimes of Eq. 5.2; `float64` accumulator is
safe (worst-case sum `≤ n-1 ≈ 10⁶`, float64 relative precision `≈ 10⁻¹⁰`).

### 3.3 Integration points

**One main-path callsite.** `Thesis/emptiness/approx/hybrid/are_seg/seg_are.go`:

```diff
- func newSegARE(keys []uint64, keyBits, K uint32, rangeLen, eps uint64,
-     backend exact.Variant) (*SegARE, error) {
+ func newSegARE(keys []uint64, keyBits, K uint32, rangeLen, segDelta uint64,
+     fprEpsilon float64, backend exact.Variant) (*SegARE, error) {
      ...
-     segs, fallbackKeys := detectSegments(keys, eps)
+     segs, fallbackKeys := detectSegments(keys, segDelta)
      ...
      fb, err := hybridutil.BuildFallback(fallbackKeys, keyBits, rangeLen, Kfb,
-         hybridutil.FallbackAlwaysSODA{}, backend)
+         hybridutil.FallbackInGapFPR{Epsilon: fprEpsilon}, backend)
```

- `NewSegARE(...,epsilon)` (main entry point) — already receives `epsilon`,
  threads it through to `newSegARE` and from there into the policy.
- `NewSegAREFromK` and `NewSegAREFromKWithBackend` (test helpers, used by
  HybridScanARE plumbing) — keep `FallbackAlwaysSODA{}` as default. Do not
  invent an `ε_eff = n·L/2^K` reconstruction.
- Rename the `eps uint64` parameter to `segDelta uint64` in `newSegARE`
  signature to remove the terminological clash with the FPR `ε float64`.

**HybridScanARE / older hybrid filters** are unaffected — they accept a
`FallbackPolicy` explicitly via `ConfigWithPolicy`, so callers can pass
`FallbackInGapFPR{ε}` if/when desired.

### 3.4 Unit tests

New file: `Thesis/emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go`.

Table-driven tests of `FallbackInGapFPR{ε}.useTrunc(keys, K, L)` against
manually-computed expected results from Eq. 5.2:

| Case | Setup | Expected |
|---|---|---|
| `empty` | n=0 | true |
| `one_key` | n=1 | true |
| `zero_spread` | all equal | true |
| `fits_K_bits` | spreadBits ≤ K | true (exact mode) |
| `dense_R_le_L` | all gaps ≤ L | true (every gap absorbs query) |
| `uniform_safe` | uniform gaps, P/(R−L) < ε | true |
| `uniform_unsafe` | uniform gaps, P/(R−L) > ε | false |
| `saturated_5pct` | 5% gaps in the saturated regime, ε = 0.01 | false (mean ≥ 0.05) |
| `saturated_5pct_high_eps` | same data, ε = 0.1 | true |
| `osm_like_safe` | huge spread, all gaps ≫ L+P | true |
| `tiny_P_safe` | P=1, gaps large | true |

Plus a **sanity test** that exercises the predicate against the actual TruncARE
FPR measured on synthetic in-gap queries:

```go
func TestFallbackInGapFPR_PredictsMeasured(t *testing.T) {
    keys := mixedKeys(1<<14, 100)
    K := uint32(20); L := uint64(128)
    predicted := computeInGapFPR(keys, K, L)
    trunc, _ := are_trunc.NewTruncAREFromK(keys, 64, K)
    weights := querygen.SmartMixWeights{NearKey: 0, InGap: 1, Uniform: 0}
    queries := querygen.GenerateSmartQueriesWeighted(keys, 1e5, L, weights, rng)
    measured := measureFPR(trunc, queries)
    // Tolerance: 10% relative or 1e-4 absolute floor (whichever is larger).
    require.InDelta(t, predicted, measured, math.Max(0.1*predicted, 1e-4))
}
```

If `computeInGapFPR` needs to live outside `useTrunc` for reuse, extract it
into an unexported helper in the same file.

### 3.5 Benchmarks (success criterion = envelope match)

New file: `bench/analysis/sosd/seg_fallback_policy_test.go`.

**Distributions:** FB, Wiki, OSM, Books, **uniform synthetic**.
**N:** 2²⁰ (smoke), 2²² (full).
**L:** 128.
**ε sweep:** {10⁻⁴, 10⁻³, 10⁻², 5·10⁻²}.
**Queries:** smart-mix (50% near-key + 30% in-gap + 20% uniform), 10⁵.

**Three lines per tradeoff plot:**
1. SegARE + `FallbackAlwaysSODA`
2. SegARE + `FallbackAlwaysTrunc`
3. SegARE + `FallbackInGapFPR{ε}` ← new

**Success criterion (envelope match).** For every (distribution, ε):

> The point `(BPK, FPR)` of the new policy must coincide with whichever of
> {AlwaysSODA, AlwaysTrunc} is on the lower envelope.

If the fallback key set is empty (FB / Wiki / Books — DBSCAN absorbs everything
into exact-mode segments), the policy is not exercised; the three lines must
coincide trivially. The test logs this and skips the comparison.

Otherwise (OSM, uniform synthetic):

- **Uniform synthetic** ⇒ `InGapFPR` must select Trunc ⇒ line ≡ Always-Trunc.
- **OSM** ⇒ `InGapFPR` must select SODA ⇒ line ≡ Always-SODA.

```go
func TestSegFallbackPolicy_MatchesEnvelope(t *testing.T) {
    for _, d := range distributions {
        for _, ε := range epsSweep {
            sodaBPK, sodaFPR := build(d, AlwaysSODA{}, ε)
            truncBPK, truncFPR := build(d, AlwaysTrunc{}, ε)
            newBPK, newFPR := build(d, InGapFPR{ε}, ε)

            if nFallback(d, ε) == 0 {
                t.Logf("%s ε=%g: empty fallback, policy not exercised", d, ε)
                continue
            }

            soda_match := within(newBPK, sodaBPK, 1.0) && within(newFPR, sodaFPR, 0.2)
            trunc_match := within(newBPK, truncBPK, 1.0) && within(newFPR, truncFPR, 0.2)

            if !soda_match && !trunc_match {
                t.Errorf("%s ε=%g: InGapFPR (BPK=%g, FPR=%g) matches neither "+
                    "SODA (%g, %g) nor Trunc (%g, %g)",
                    d, ε, newBPK, newFPR, sodaBPK, sodaFPR, truncBPK, truncFPR)
            }
        }
    }
}
```

**Artifacts:**
- CSV: `bench_results/data/seg_fallback_policy.csv`
- SVG: `bench_results/plots/seg_fallback_policy_<dist>.svg`
- PNG (rendered from SVG): for visual review per the global memo about always
  inspecting SVGs visually.

**Run order:** smoke first (uniform-only — clearest signal, Trunc should win),
then full sweep. Clean `bench_results/{data,plots}/seg_fallback_policy*`
between runs.

## 4. Out of scope

- Changing the `FallbackPolicy` interface.
- Touching `HybridScanARE` / `ConfigWithPolicy` callers.
- `ε`-reconstruction in `NewSegAREFromK*` test helpers.
- Tuning `t` inside `AdaptiveARE` (clean negative answer — see §2.4).
- Outlier-protected `spread` for `TruncARE` (a separate filter-level concern).
- Worst-case variant `FallbackInGapFPRWorst{ε}` (`min R_i > L + P/ε`). May be
  added in a follow-up if the mean variant proves insufficient in the bench.
- A new thesis section describing the policy and proof. Will be written
  separately after empirical validation.

## 5. Risks

1. **Mixed-bucket workload**: the formula bounds the 30% in-gap fraction only.
   The other 70% (near-key + uniform) have FPR bounded by trunc-on-shorter-queries
   or by gap-formula on a randomly-chosen gap, both ≤ the modelled in-gap FPR. So
   the policy is conservative on the full mix; the bench should confirm this.
2. **Continuous-approximation slop** (`2^t` vs `2^t - 1`): negligible for `K ≥ 4`;
   may show up as 1-bit-of-noise rejections at very small K. Bench will reveal.
3. **DBSCAN absorbs everything on FB/Wiki/Books** ⇒ empty fallback ⇒ policy
   never invoked on those datasets. Tests skip cleanly in that case; not a defect.
