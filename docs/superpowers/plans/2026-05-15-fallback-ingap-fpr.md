# FallbackInGapFPR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unconditional `FallbackAlwaysSODA{}` policy used by `SegARE` with a data-aware `FallbackInGapFPR{ε}` policy that directly evaluates the proven in-gap FPR formula from thesis Section 5.2.

**Architecture:** New `FallbackPolicy` implementation in `hybridutil` evaluates `(1/(n−1)) · Σ min(1, P/(R_i − L)) ≤ ε` in an `O(n)` pass over the sorted fallback keys, where `P = spread >> K`. The `newSegARE` private constructor is refactored to take a `FallbackPolicy` directly; `NewSegARE` (which has `epsilon`) constructs the new policy, while `NewSegAREFromK*` test-helper constructors keep `FallbackAlwaysSODA{}`. Validity is checked empirically via an envelope-matching benchmark on SOSD distributions.

**Tech Stack:** Go 1.21+, standard library only. Tests use `testing.T` table-driven style. Bench uses existing `bench/analysis/sosd` SVG-tradeoff machinery.

**Spec:** `docs/superpowers/specs/2026-05-15-fallback-ingap-fpr-design.md`

**Commit style (from project `CLAUDE.md`):** `feat(scope):`, `test(scope):`, `refactor(scope):`, `bench(scope):`. No `Co-Authored-By` lines.

---

## File map

| File | Action | Purpose |
|---|---|---|
| `Thesis/emptiness/approx/hybrid/hybridutil/policy.go` | Modify | Add `FallbackInGapFPR` struct, `useTrunc`, `inGapFPRMean` helper |
| `Thesis/emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go` | Create | Table tests + predicts-measured sanity test |
| `Thesis/emptiness/approx/hybrid/are_seg/seg_are.go` | Modify | Refactor `newSegARE` signature to take `FallbackPolicy`; switch `NewSegARE` to `FallbackInGapFPR` |
| `bench/analysis/sosd/seg_fallback_policy_test.go` | Create | Envelope benchmark on SOSD + uniform |

## Working directory

All work happens in worktree: `/Users/andrei.ogurtsov/Thesis-Bench-industry/.claude/worktrees/fallback-ingap-fpr`

All `cd Thesis` commands below assume CWD is the worktree root.

---

### Task 1: `FallbackInGapFPR` skeleton — edge cases only

**Files:**
- Modify: `Thesis/emptiness/approx/hybrid/hybridutil/policy.go`
- Create: `Thesis/emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go`

#### - [ ] Step 1: Write failing edge-case tests

Create `Thesis/emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go`:

```go
package hybridutil

import "testing"

func TestFallbackInGapFPR_EdgeCases(t *testing.T) {
	tests := []struct {
		name string
		keys []uint64
		K    uint32
		L    uint64
		eps  float64
		want bool
	}{
		{"empty", []uint64{}, 16, 8, 1e-3, true},
		{"one_key", []uint64{42}, 16, 8, 1e-3, true},
		{"zero_spread", []uint64{5, 5, 5}, 16, 8, 1e-3, true},
		// spread = 300, spreadBits = 9; K = 32 → spreadBits ≤ K → exact mode → true
		{"fits_K_bits", []uint64{0, 100, 200, 300}, 32, 8, 1e-3, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := FallbackInGapFPR{Epsilon: tc.eps}.useTrunc(tc.keys, tc.K, tc.L)
			if got != tc.want {
				t.Errorf("useTrunc=%v want=%v", got, tc.want)
			}
		})
	}
}

func TestFallbackInGapFPR_String(t *testing.T) {
	if got := (FallbackInGapFPR{Epsilon: 1e-3}).String(); got != "InGapFPR" {
		t.Errorf("String()=%q want %q", got, "InGapFPR")
	}
}
```

#### - [ ] Step 2: Verify the tests fail (compile error — type doesn't exist)

Run:
```bash
cd Thesis && go test ./emptiness/approx/hybrid/hybridutil/ 2>&1 | head -10
```
Expected: compile error `undefined: FallbackInGapFPR`.

#### - [ ] Step 3: Add skeleton struct + edge-case branches to `policy.go`

Open `Thesis/emptiness/approx/hybrid/hybridutil/policy.go`. The file already imports `mbits "math/bits"`. Append at the end of the file:

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

func (f FallbackInGapFPR) useTrunc(keys []uint64, K uint32, L uint64) bool {
	n := len(keys)
	if n < 2 {
		return true
	}
	spread := keys[n-1] - keys[0]
	if spread == 0 {
		return true
	}
	spreadBits := uint32(64 - mbits.LeadingZeros64(spread))
	if spreadBits <= K {
		return true // exact mode inside TruncARE → FPR = 0
	}
	// Per-gap loop arrives in Task 2; reject conservatively for now.
	return false
}

func (FallbackInGapFPR) String() string { return "InGapFPR" }
```

#### - [ ] Step 4: Verify tests pass

Run:
```bash
cd Thesis && go test -v -run TestFallbackInGapFPR ./emptiness/approx/hybrid/hybridutil/
```
Expected: `PASS`, 5 subtests (4 edge cases + String).

#### - [ ] Step 5: Commit

```bash
cd Thesis && git add emptiness/approx/hybrid/hybridutil/policy.go \
    emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go
git commit -m "feat(hybridutil): add FallbackInGapFPR skeleton with edge-case handling"
```

---

### Task 2: Implement the per-gap formula (Eq. 5.2)

**Files:**
- Modify: `Thesis/emptiness/approx/hybrid/hybridutil/policy.go`
- Modify: `Thesis/emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go`

#### - [ ] Step 1: Add uniform-keys tests for both verdicts

Append to `policy_ingapfpr_test.go` (above any existing functions, after imports):

```go
// uniformKeys builds n keys with constant gap R, starting from 0.
func uniformKeys(n int, R uint64) []uint64 {
	out := make([]uint64, n)
	for i := range out {
		out[i] = uint64(i) * R
	}
	return out
}

func TestFallbackInGapFPR_Uniform(t *testing.T) {
	// n = 1024, R = 1<<20 → spread = 1023<<20. With K = 12:
	//   P = spread >> 12  ≈ 1023<<8 ≈ 261888
	//   per-gap FPR ≈ P/(R-L) = 261888/(1048576-128) ≈ 0.25
	// → unsafe at ε=0.01
	t.Run("unsafe_K12_R1M_L128", func(t *testing.T) {
		keys := uniformKeys(1024, 1<<20)
		got := FallbackInGapFPR{Epsilon: 0.01}.useTrunc(keys, 12, 128)
		if got {
			t.Errorf("useTrunc=true want false (P/(R-L)≈0.25 ≫ 0.01)")
		}
	})

	// With K = 24:
	//   P = spread >> 24 ≈ 1023>>4 ≈ 63
	//   per-gap FPR ≈ 63/(1048576-128) ≈ 6.0e-5
	// → safe at ε=1e-3
	t.Run("safe_K24_R1M_L128", func(t *testing.T) {
		keys := uniformKeys(1024, 1<<20)
		got := FallbackInGapFPR{Epsilon: 1e-3}.useTrunc(keys, 24, 128)
		if !got {
			t.Errorf("useTrunc=false want true (P/(R-L)≈6e-5 ≪ 1e-3)")
		}
	})
}
```

#### - [ ] Step 2: Verify the new tests fail

Run:
```bash
cd Thesis && go test -v -run TestFallbackInGapFPR_Uniform ./emptiness/approx/hybrid/hybridutil/
```
Expected: `safe_K24_R1M_L128` FAILS (returns false because formula is stubbed); `unsafe` passes (returns false coincidentally).

#### - [ ] Step 3: Implement the per-gap loop in `policy.go`

Replace the stub `return false` in `FallbackInGapFPR.useTrunc` with the full formula. Final method:

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
	spreadBits := uint32(64 - mbits.LeadingZeros64(spread))
	if spreadBits <= K {
		return true // exact mode inside TruncARE → FPR = 0
	}
	P := spread >> K
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
			sum += 1.0 // saturated: P/(R-L) ≥ 1
		default:
			sum += float64(P) / float64(R-L) // Eq. 5.2
		}
	}
	return sum/float64(n-1) <= f.Epsilon
}
```

#### - [ ] Step 4: Verify all tests pass

Run:
```bash
cd Thesis && go test -v -run TestFallbackInGapFPR ./emptiness/approx/hybrid/hybridutil/
```
Expected: all subtests (edge cases + uniform safe/unsafe + String) PASS.

#### - [ ] Step 5: Commit

```bash
cd Thesis && git add emptiness/approx/hybrid/hybridutil/policy.go \
    emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go
git commit -m "feat(hybridutil): implement FallbackInGapFPR per-gap FPR formula"
```

---

### Task 3: Saturated, dense, and OSM-like edge tests

**Files:**
- Modify: `Thesis/emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go`

Implementation already passes; this task only widens test coverage.

#### - [ ] Step 1: Add helpers and tests

Append to `policy_ingapfpr_test.go`:

```go
// mixedKeys builds nLarge keys with gap = largeGap, then appends nSmall keys
// packed tightly at the end (gap = 1). Useful for "X% saturated gaps" cases.
func mixedKeys(nLarge int, largeGap uint64, nSmall int) []uint64 {
	out := make([]uint64, 0, nLarge+nSmall)
	for i := 0; i < nLarge; i++ {
		out = append(out, uint64(i)*largeGap)
	}
	base := uint64(nLarge) * largeGap
	for i := 0; i < nSmall; i++ {
		out = append(out, base+uint64(i+1))
	}
	return out
}

func TestFallbackInGapFPR_Saturated(t *testing.T) {
	// 1000 large gaps (1<<20 each), then 50 tiny gaps (=1). L=16, K chosen so
	// large gaps are safe (P/(R-L) ≪ 1) but tiny gaps saturate (R-L < 0 → FPR=0
	// actually). To force *saturated* we need R > L but R-L ≤ P. Use L=0 and
	// gap=1: R-L=1; P = spread >> K. With nLarge=1000 large gaps ≫ 50 small
	// ones, spread ≈ nLarge·largeGap ≈ 2^30. K=10 → P=2^20 ≫ 1 → small gaps
	// saturate (sum += 1 each).
	keys := mixedKeys(1000, 1<<20, 50)
	// 50 saturated out of 1049 gaps; mean ≈ 50/1049 ≈ 0.0477.
	t.Run("eps_001_reject", func(t *testing.T) {
		got := FallbackInGapFPR{Epsilon: 0.01}.useTrunc(keys, 10, 0)
		if got {
			t.Errorf("useTrunc=true want false (mean FPR ≈ 0.048 > 0.01)")
		}
	})
	t.Run("eps_010_accept", func(t *testing.T) {
		got := FallbackInGapFPR{Epsilon: 0.10}.useTrunc(keys, 10, 0)
		if !got {
			t.Errorf("useTrunc=false want true (mean FPR ≈ 0.048 < 0.10)")
		}
	})
}

func TestFallbackInGapFPR_DenseGapsLE_L(t *testing.T) {
	// All gaps = 4 ≤ L = 8 → every gap contributes FPR_i = 0 → safe.
	keys := uniformKeys(1024, 4)
	got := FallbackInGapFPR{Epsilon: 1e-9}.useTrunc(keys, 12, 8)
	if !got {
		t.Errorf("useTrunc=false want true (all gaps ≤ L → sum=0)")
	}
}

func TestFallbackInGapFPR_HugeSparseSpread(t *testing.T) {
	// OSM-like: spread = 2^59, n = 2^16 → typical gap ≈ 2^43.
	// K = 37 → P ≈ 2^22. Per-gap FPR ≈ 2^22 / (2^43 - L) ≈ 2^-21 ≈ 5e-7.
	keys := uniformKeys(1<<16, 1<<43)
	got := FallbackInGapFPR{Epsilon: 1e-3}.useTrunc(keys, 37, 128)
	if !got {
		t.Errorf("useTrunc=false want true (huge sparse gaps → FPR ≪ ε)")
	}
}
```

#### - [ ] Step 2: Run all hybridutil tests

```bash
cd Thesis && go test -v -run TestFallbackInGapFPR ./emptiness/approx/hybrid/hybridutil/
```
Expected: all subtests PASS (edge + uniform + saturated + dense + huge-sparse).

#### - [ ] Step 3: Commit

```bash
cd Thesis && git add emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go
git commit -m "test(hybridutil): saturated, dense, and OSM-like cases for FallbackInGapFPR"
```

---

### Task 4: Sanity test — predicted FPR matches measured

**Files:**
- Modify: `Thesis/emptiness/approx/hybrid/hybridutil/policy.go` (extract helper)
- Modify: `Thesis/emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go`

#### - [ ] Step 1: Extract `inGapFPRMean` helper, reuse from `useTrunc`

In `Thesis/emptiness/approx/hybrid/hybridutil/policy.go`, refactor `FallbackInGapFPR.useTrunc` to delegate. Replace the method body with:

```go
func (f FallbackInGapFPR) useTrunc(keys []uint64, K uint32, L uint64) bool {
	mean, ok := inGapFPRMean(keys, K, L)
	if !ok {
		return true // edge case: trivially safe (n<2, spread=0, or exact-mode)
	}
	return mean <= f.Epsilon
}

// inGapFPRMean returns (1/(n-1)) · Σ min(1, P/(R_i - L)) for the sorted keys.
// ok=false signals an edge case where TruncARE is trivially safe:
//   - n < 2
//   - spread = 0
//   - spread fits in K bits (exact mode inside TruncARE)
// In all edge cases the returned mean is 0.
func inGapFPRMean(keys []uint64, K uint32, L uint64) (mean float64, ok bool) {
	n := len(keys)
	if n < 2 {
		return 0, false
	}
	spread := keys[n-1] - keys[0]
	if spread == 0 {
		return 0, false
	}
	spreadBits := uint32(64 - mbits.LeadingZeros64(spread))
	if spreadBits <= K {
		return 0, false
	}
	P := spread >> K
	if P == 0 {
		P = 1
	}
	var sum float64
	for i := 1; i < n; i++ {
		R := keys[i] - keys[i-1]
		switch {
		case R <= L:
			// FPR_i = 0
		case R-L <= P:
			sum += 1.0
		default:
			sum += float64(P) / float64(R-L)
		}
	}
	return sum / float64(n-1), true
}
```

#### - [ ] Step 2: Verify existing tests still pass

```bash
cd Thesis && go test -v -run TestFallbackInGapFPR ./emptiness/approx/hybrid/hybridutil/
```
Expected: all subtests PASS.

#### - [ ] Step 3: Add sanity test that measures actual FPR

Append to `policy_ingapfpr_test.go` (this requires `are_trunc` import — add it to the existing import block at the top of the file):

Update the import block at the top of `policy_ingapfpr_test.go` to:

```go
package hybridutil

import (
	"math"
	"math/rand"
	"testing"

	"Thesis/emptiness/approx/are_trunc"
)
```

Then append:

```go
// generateInGapQueries mirrors the in-gap branch of
// bench/internal/querygen.GenerateSmartQueriesWeighted: pick a gap uniformly
// by index, then place the query start uniformly inside that gap.
func generateInGapQueries(keys []uint64, count int, L uint64, rng *rand.Rand) [][2]uint64 {
	n := len(keys)
	if n < 2 {
		return nil
	}
	type gap struct{ lo, hi uint64 }
	gaps := make([]gap, 0, n-1)
	for i := 0; i < n-1; i++ {
		if keys[i+1]-keys[i] > 1 {
			gaps = append(gaps, gap{keys[i] + 1, keys[i+1] - 1})
		}
	}
	if len(gaps) == 0 {
		return nil
	}
	out := make([][2]uint64, 0, count)
	for attempts := 0; attempts < count*4 && len(out) < count; attempts++ {
		g := gaps[rng.Intn(len(gaps))]
		gapLen := g.hi - g.lo + 1
		if gapLen == 0 {
			continue
		}
		a := g.lo + uint64(rng.Int63n(int64(gapLen)))
		b := a + L - 1
		if b > g.hi {
			b = g.hi
		}
		if b >= a {
			out = append(out, [2]uint64{a, b})
		}
	}
	return out
}

func measureFPR(t *testing.T, f *are_trunc.TruncARE, queries [][2]uint64) float64 {
	t.Helper()
	if len(queries) == 0 {
		t.Fatal("no queries to measure")
	}
	fp := 0
	for _, q := range queries {
		if !f.IsEmpty(q[0], q[1]) {
			fp++
		}
	}
	return float64(fp) / float64(len(queries))
}

func TestFallbackInGapFPR_PredictsMeasured(t *testing.T) {
	// 1<<14 keys, uniform gap = 1<<20, L = 128, K = 24.
	// P/(R-L) ≈ 63/(2^20 - 128) ≈ 6e-5 → very low FPR but non-zero.
	const n = 1 << 14
	const R = uint64(1) << 20
	const L = uint64(128)
	const K = uint32(24)
	keys := uniformKeys(n, R)
	const keyBits uint32 = 64

	predicted, ok := inGapFPRMean(keys, K, L)
	if !ok {
		t.Fatalf("expected non-trivial mean, got edge-case")
	}

	trunc, err := are_trunc.NewTruncAREFromK(keys, keyBits, K)
	if err != nil {
		t.Fatalf("TruncARE build: %v", err)
	}
	rng := rand.New(rand.NewSource(42))
	queries := generateInGapQueries(keys, 100_000, L, rng)
	measured := measureFPR(t, trunc, queries)

	// Tolerance: 30% relative or 5e-4 absolute floor (whichever is larger).
	// Wider than the 10% from the spec because the in-gap empirical FPR has
	// non-trivial Monte-Carlo variance at low predicted values.
	tol := math.Max(0.30*predicted, 5e-4)
	if math.Abs(predicted-measured) > tol {
		t.Errorf("predicted=%g measured=%g (|diff|=%g > tol=%g)",
			predicted, measured, math.Abs(predicted-measured), tol)
	}
	t.Logf("predicted=%g measured=%g diff=%g tol=%g",
		predicted, measured, math.Abs(predicted-measured), tol)
}
```

#### - [ ] Step 4: Run the sanity test

```bash
cd Thesis && go test -v -run TestFallbackInGapFPR_PredictsMeasured ./emptiness/approx/hybrid/hybridutil/
```
Expected: PASS. Log line shows `predicted` and `measured` within `tol`.

If `are_trunc.NewTruncAREFromK` signature differs from `(keys, keyBits, K) → (*TruncARE, error)`, inspect the file:
```bash
grep -n "^func NewTruncAREFromK" Thesis/emptiness/approx/are_trunc/*.go
```
and adapt the call accordingly. The test should still measure FPR by calling `IsEmpty(lo, hi)`.

#### - [ ] Step 5: Commit

```bash
cd Thesis && git add emptiness/approx/hybrid/hybridutil/policy.go \
    emptiness/approx/hybrid/hybridutil/policy_ingapfpr_test.go
git commit -m "test(hybridutil): predicts-measured sanity check + inGapFPRMean helper"
```

---

### Task 5: Refactor `newSegARE` to take `FallbackPolicy` directly

This task is **pure refactoring** — no behavior change. `NewSegARE`, `NewSegAREFromK`, and `NewSegAREFromKWithBackend` all keep using `FallbackAlwaysSODA{}`. Only the internal signature changes.

**Files:**
- Modify: `Thesis/emptiness/approx/hybrid/are_seg/seg_are.go`

#### - [ ] Step 1: Verify existing SegARE tests pass (baseline)

```bash
cd Thesis && go test ./emptiness/approx/hybrid/are_seg/
```
Expected: `ok` for the package.

#### - [ ] Step 2: Refactor `newSegARE` signature

Open `Thesis/emptiness/approx/hybrid/are_seg/seg_are.go`. Find the function `newSegARE` (currently at line 51) with signature:

```go
func newSegARE(keys []uint64, keyBits, K uint32, rangeLen, eps uint64, backend exact.Variant) (*SegARE, error) {
```

Replace it with:

```go
func newSegARE(keys []uint64, keyBits, K uint32, rangeLen, segDelta uint64, policy hybridutil.FallbackPolicy, backend exact.Variant) (*SegARE, error) {
```

Inside the function body:

- Replace the local use of `eps` (the DBSCAN δ-radius, NOT the FPR ε) with `segDelta`. The single line affected is:
  ```go
  segs, fallbackKeys := detectSegments(keys, eps)
  ```
  becomes:
  ```go
  segs, fallbackKeys := detectSegments(keys, segDelta)
  ```

- Replace the hardcoded fallback policy. The current line (around line 86):
  ```go
  fb, err := hybridutil.BuildFallback(fallbackKeys, keyBits, rangeLen, Kfb, hybridutil.FallbackAlwaysSODA{}, backend)
  ```
  becomes:
  ```go
  fb, err := hybridutil.BuildFallback(fallbackKeys, keyBits, rangeLen, Kfb, policy, backend)
  ```

#### - [ ] Step 3: Update three callers (still pass `FallbackAlwaysSODA{}` — no behavior change yet)

In the same file, update the three places that call `newSegARE`:

**(a) `NewSegARE` (around line 36):** The line
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, exact.VariantAuto)
```
becomes:
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, hybridutil.FallbackAlwaysSODA{}, exact.VariantAuto)
```

**(b) `NewSegAREFromK` (around line 136):** The line
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, exact.VariantAuto)
```
becomes:
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, hybridutil.FallbackAlwaysSODA{}, exact.VariantAuto)
```

**(c) `NewSegAREFromKWithBackend` (around line 167):** The line
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, backend)
```
becomes:
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, hybridutil.FallbackAlwaysSODA{}, backend)
```

#### - [ ] Step 4: Verify behavior unchanged

```bash
cd Thesis && go test ./emptiness/approx/hybrid/are_seg/ ./emptiness/approx/hybrid/hybridutil/
```
Expected: `ok` for both packages, no regressions.

#### - [ ] Step 5: Commit

```bash
cd Thesis && git add emptiness/approx/hybrid/are_seg/seg_are.go
git commit -m "refactor(are_seg): thread FallbackPolicy through newSegARE"
```

---

### Task 6: Switch `NewSegARE` to `FallbackInGapFPR{epsilon}`

`NewSegARE` already receives the FPR `epsilon float64`. Wire it into the new policy. `NewSegAREFromK*` test helpers stay on `FallbackAlwaysSODA{}` (no `ε` available).

**Files:**
- Modify: `Thesis/emptiness/approx/hybrid/are_seg/seg_are.go`

#### - [ ] Step 1: Change `NewSegARE` to construct `FallbackInGapFPR`

In `Thesis/emptiness/approx/hybrid/are_seg/seg_are.go`, find `NewSegARE` (around line 36). The current line:
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, hybridutil.FallbackAlwaysSODA{}, exact.VariantAuto)
```
becomes:
```go
return newSegARE(keys, keyBits, K, rangeLen, eps, hybridutil.FallbackInGapFPR{Epsilon: epsilon}, exact.VariantAuto)
```

Leave `NewSegAREFromK` and `NewSegAREFromKWithBackend` untouched (they keep `FallbackAlwaysSODA{}`).

#### - [ ] Step 2: Run SegARE tests

```bash
cd Thesis && go test -v ./emptiness/approx/hybrid/are_seg/ 2>&1 | tail -30
```

Expected: all tests PASS. The cluster_stats_test and seg_are_test should still work — they go through `NewSegARE` or `NewSegAREFromK`. If any test fails because it was implicitly relying on the AlwaysSODA fallback for a specific FPR/BPK outcome, **do not silently mute it**: read the test, understand what it asserts, and decide whether (a) the assertion is wrong now and needs updating to reflect the new policy, or (b) the new policy is incorrect.

#### - [ ] Step 3: Run wider hybrid tests to catch downstream impact

```bash
cd Thesis && go test ./emptiness/approx/hybrid/... 2>&1 | tail
```

Expected: all `ok`. If `are_dbscan` or others break, they should not — they use their own `newHybridScanARE`, not `newSegARE`. But verify.

#### - [ ] Step 4: Commit

```bash
cd Thesis && git add emptiness/approx/hybrid/are_seg/seg_are.go
git commit -m "feat(are_seg): switch NewSegARE fallback to FallbackInGapFPR"
```

---

### Task 7: Envelope benchmark on SOSD + uniform

Validates the success criterion from the spec §3.5: the `FallbackInGapFPR` line must coincide with whichever of {AlwaysSODA, AlwaysTrunc} is on the lower envelope of (FPR, BPK).

**Files:**
- Create: `bench/analysis/sosd/seg_fallback_policy_test.go`

Bench tests live in the root module (not the Thesis submodule), so they CAN import both `Thesis/...` packages and `bench/internal/querygen`.

#### - [ ] Step 1: Read existing bench template for reference

```bash
cat /Users/andrei.ogurtsov/Thesis-Bench-industry/bench/analysis/sosd/tradeoff_osm_test.go | head -200
```

Note: where it loads SOSD keys (`testutils.GetBenchKeys` or `loadSOSDUint64` from `bench/config_test.go`), how it constructs queries via `querygen.GenerateSmartQueriesWeighted`, how it computes FPR/BPK, and how it calls `testutils.GenerateTradeoffSVG`. Adopt the same machinery — do NOT reinvent.

#### - [ ] Step 2: Write the benchmark test

Create `bench/analysis/sosd/seg_fallback_policy_test.go`:

```go
package sosd_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"Thesis/emptiness/approx/hybrid/are_seg"
	"Thesis/emptiness/approx/hybrid/hybridutil"

	"Thesis-bench-industry/bench/internal/querygen"
)

// withinAbs reports whether |a-b| <= tol.
func withinAbs(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// withinRel reports whether |a-b| <= max(absTol, relTol*max(|a|,|b|)).
func withinRel(a, b, absTol, relTol float64) bool {
	scale := math.Max(math.Abs(a), math.Abs(b))
	return math.Abs(a-b) <= math.Max(absTol, relTol*scale)
}

type segRunResult struct {
	policy    string
	bpk       float64
	fpr       float64
	nClusters int
	nFallback int
}

func runSegOnce(t *testing.T, keys []uint64, queries [][2]uint64, L uint64, eps float64, policy hybridutil.FallbackPolicy, policyName string) segRunResult {
	t.Helper()
	// Use the from-K constructor so the policy can be passed explicitly.
	// K = ceil(log2(n·L/ε)); reuse the same formula used by NewSegARE.
	n := len(keys)
	K := uint32(math.Ceil(math.Log2(float64(n) * (float64(L) + 1) / eps)))
	if K == 0 {
		K = 1
	}
	if K > 64 {
		K = 64
	}
	// NewSegAREFromK uses FallbackAlwaysSODA by default. To compare three
	// policies on identical δ/K we instead use the public NewSegARE for the
	// InGapFPR run (which threads ε), and we accept that AlwaysSODA/
	// AlwaysTrunc runs use the same NewSegAREFromK + explicit policy via a
	// thin local helper. Since the only difference between the three runs is
	// the fallback policy, we construct SegARE directly here.
	//
	// However, are_seg currently exposes only NewSegARE (uses InGapFPR after
	// Task 6) and NewSegAREFromK (uses AlwaysSODA). To switch policies cleanly
	// in this bench we need a constructor that accepts a policy. If one is
	// missing, add NewSegAREFromKWithPolicy (mirrors NewSegAREFromKWithBackend
	// but takes a hybridutil.FallbackPolicy). See note below the test body.
	_ = K
	_ = policy
	_ = policyName

	t.Skipf("TODO: switch to NewSegAREFromKWithPolicy once the constructor exists; see file note")
	return segRunResult{}
}

func TestSegFallbackPolicy_Smoke_Uniform(t *testing.T) {
	const n = 1 << 20
	const L = uint64(128)
	const eps = 1e-3
	const gap = uint64(1) << 30 // truly uniform → InGapFPR should choose Trunc

	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = uint64(i) * gap
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	rng := rand.New(rand.NewSource(42))
	queries := querygen.GenerateSmartQueriesWeighted(keys, 100_000, L,
		querygen.SmartMixWeights{
			NearKey: querygen.QueryWeightNearKey,
			InGap:   querygen.QueryWeightInGap,
			Uniform: querygen.QueryWeightUniform,
		}, rng)

	soda := runSegOnce(t, keys, queries, L, eps, hybridutil.FallbackAlwaysSODA{}, "AlwaysSODA")
	trunc := runSegOnce(t, keys, queries, L, eps, hybridutil.FallbackAlwaysTrunc{}, "AlwaysTrunc")
	inGap := runSegOnce(t, keys, queries, L, eps, hybridutil.FallbackInGapFPR{Epsilon: eps}, "InGapFPR")

	t.Logf("uniform ε=%g: SODA=(BPK=%.2f, FPR=%.3e)  Trunc=(BPK=%.2f, FPR=%.3e)  InGapFPR=(BPK=%.2f, FPR=%.3e)",
		eps, soda.bpk, soda.fpr, trunc.bpk, trunc.fpr, inGap.bpk, inGap.fpr)

	// Envelope check: on uniform data Trunc is the lower envelope.
	if inGap.nFallback == 0 {
		t.Logf("empty fallback — policy not exercised, skipping envelope check")
		return
	}
	sodaMatch := withinAbs(inGap.bpk, soda.bpk, 1.0) && withinRel(inGap.fpr, soda.fpr, 5e-4, 0.30)
	truncMatch := withinAbs(inGap.bpk, trunc.bpk, 1.0) && withinRel(inGap.fpr, trunc.fpr, 5e-4, 0.30)
	if !sodaMatch && !truncMatch {
		t.Errorf("InGapFPR does not match either reference line")
	}
	if !truncMatch {
		t.Errorf("uniform: expected InGapFPR ≈ Trunc envelope, got match=%v (Trunc=(%.2f,%.3e) vs InGapFPR=(%.2f,%.3e))",
			truncMatch, trunc.bpk, trunc.fpr, inGap.bpk, inGap.fpr)
	}

	// Persist CSV row for later aggregation.
	persistRow(t, "uniform", eps, soda, trunc, inGap)
}

func persistRow(t *testing.T, dist string, eps float64, soda, trunc, inGap segRunResult) {
	t.Helper()
	outDir := filepath.Join("..", "..", "..", "bench_results", "data")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Logf("mkdir %s: %v", outDir, err)
		return
	}
	path := filepath.Join(outDir, "seg_fallback_policy.csv")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Logf("open csv: %v", err)
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "%s,%g,%s,%.4f,%.6g,%d,%d\n",
		dist, eps, soda.policy, soda.bpk, soda.fpr, soda.nClusters, soda.nFallback)
	fmt.Fprintf(f, "%s,%g,%s,%.4f,%.6g,%d,%d\n",
		dist, eps, trunc.policy, trunc.bpk, trunc.fpr, trunc.nClusters, trunc.nFallback)
	fmt.Fprintf(f, "%s,%g,%s,%.4f,%.6g,%d,%d\n",
		dist, eps, inGap.policy, inGap.bpk, inGap.fpr, inGap.nClusters, inGap.nFallback)
}
```

**Note inside the file body:** `runSegOnce` currently calls `t.Skipf` because `are_seg` does not expose a constructor that takes both `K` and a `FallbackPolicy` explicitly. Add `NewSegAREFromKWithPolicy` in the next sub-step.

#### - [ ] Step 3: Add `NewSegAREFromKWithPolicy` constructor to are_seg

Open `Thesis/emptiness/approx/hybrid/are_seg/seg_are.go`. Find `NewSegAREFromKWithBackend` (around line 167). Immediately after it, append:

```go
// NewSegAREFromKWithPolicy is NewSegAREFromKWithBackend with an explicit
// fallback policy. Used by benchmarks that need to compare different policies
// on identical δ/K.
func NewSegAREFromKWithPolicy(keys []uint64, keyBits, K uint32, rangeLen uint64, policy hybridutil.FallbackPolicy, backend exact.Variant) (*SegARE, error) {
	errutil.BugOn(keyBits == 0 || keyBits > 64, "keyBits must be in [1,64], got %d", keyBits)
	errutil.BugOn(K == 0 || K > 64, "K must be in (0,64], got %d", K)

	n := len(keys)
	if n == 0 {
		return &SegARE{}, nil
	}

	var pow float64
	if K >= 64 {
		pow = float64(^uint64(0)) + 1
	} else {
		pow = float64(uint64(1) << K)
	}
	v := float64(segMinPts) * pow / float64(n)
	var eps uint64
	switch {
	case v < 1:
		eps = 1
	case v >= float64(math.MaxUint64):
		eps = math.MaxUint64
	default:
		eps = uint64(v)
	}

	return newSegARE(keys, keyBits, K, rangeLen, eps, policy, backend)
}
```

Verify the existing tests still pass:
```bash
cd Thesis && go test ./emptiness/approx/hybrid/are_seg/
```
Expected: `ok`.

#### - [ ] Step 4: Wire `runSegOnce` to actually run

Open `bench/analysis/sosd/seg_fallback_policy_test.go`. Replace the body of `runSegOnce` (the part starting at `_ = K` through the `t.Skipf`) with:

```go
	filter, err := are_seg.NewSegAREFromKWithPolicy(keys, 64, K, L, policy, exact.VariantAuto)
	if err != nil {
		t.Fatalf("SegARE build (%s): %v", policyName, err)
	}
	fp := 0
	for _, q := range queries {
		if !filter.IsEmpty(q[0], q[1]) {
			fp++
		}
	}
	fpr := float64(fp) / float64(len(queries))
	bpk := float64(filter.SizeInBits()) / float64(len(keys))
	nC, nF, _ := filter.Stats()
	return segRunResult{
		policy:    policyName,
		bpk:       bpk,
		fpr:       fpr,
		nClusters: nC,
		nFallback: nF,
	}
```

Add the missing import to the file:

```go
"Thesis/emptiness/exact"
```

#### - [ ] Step 5: Run the smoke benchmark

Clean any pre-existing cache, then run:

```bash
rm -f bench_results/data/seg_fallback_policy.csv
go test -v -timeout 30m -run TestSegFallbackPolicy_Smoke_Uniform ./bench/analysis/sosd/
```

Expected: PASS. The log line shows three (BPK, FPR) triples; the InGapFPR result must match the Trunc envelope (lower BPK with comparable FPR). If it instead matches SODA on uniform data, the policy is mis-calibrated for low-density uniform-gap distributions — investigate before proceeding.

Inspect:
```bash
cat bench_results/data/seg_fallback_policy.csv
```
Expected: 3 rows (SODA, Trunc, InGapFPR) for `uniform, eps=1e-3`.

#### - [ ] Step 6: Commit

```bash
git add Thesis/emptiness/approx/hybrid/are_seg/seg_are.go \
    bench/analysis/sosd/seg_fallback_policy_test.go
git commit -m "bench(sosd): envelope-matching benchmark for FallbackInGapFPR"
```

After Task 7 lands, the orchestrator (you, or the user) runs the smoke test, then the full SOSD sweep, collects results, and writes the final report.

---

## Self-review

**Spec coverage:**

| Spec section | Implemented by |
|---|---|
| §3.1 New policy type | Task 1 |
| §3.2 Algorithm | Task 2 |
| §3.3 Integration points (`newSegARE` refactor + main-path switch) | Tasks 5, 6 |
| §3.4 Unit tests (table + sanity) | Tasks 1–4 |
| §3.5 Envelope benchmark | Task 7 |
| §4 Out of scope | not touched |

**Placeholder scan:** none — every step contains the actual code/command.

**Type consistency:** `inGapFPRMean` is introduced in Task 4 and used in `useTrunc`; `NewSegAREFromKWithPolicy` is introduced in Task 7 Step 3 and used in Task 7 Step 4. The signature `FallbackInGapFPR{Epsilon: ε}` is consistent across Tasks 1, 2, 4, 6, 7. The `policy` parameter inserted into `newSegARE` in Task 5 keeps the same type (`hybridutil.FallbackPolicy`) throughout.

**Two minor caveats already flagged inline:**
1. Task 4 Step 4: if `are_trunc.NewTruncAREFromK` has a different signature, adapt the call (instructions provided inline).
2. Task 7 Step 5: if InGapFPR matches SODA on uniform data, do not paper over — investigate.
