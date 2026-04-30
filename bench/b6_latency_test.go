package bench_test

import (
	"Thesis-bench-industry/thirdparty/snarf"
	"Thesis-bench-industry/thirdparty/surf"
	"Thesis/emptiness/approx/are_bloom"
	"Thesis/emptiness/approx/are_greedy_scan"
	"Thesis/emptiness/approx/are_hybrid_scan"
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/emptiness/approx/are_trunc"
	"encoding/json"
	"fmt"
	"math"
	mathbits "math/bits"
	"math/rand"
	"os"
	"runtime"
	"testing"
	"time"
)

// TestB6IndustryLatency measures build throughput and query latency for the
// headline filter set on SOSD Books at n=2^24, across a few representative
// range lengths. It is the data source for the `B6` defence backup slide
// and the build/query latency table in §sec:eval-build-query-latency.
//
// Output files:
//   - bench_results/data/b6_latency.json (machine-readable)
//   - the log line stream is what the test prints; redirect via tee
//
// Filters covered (per Plan Task 1.2):
//   {Grafite, SNARF, SuRFReal(8), SODA, Truncation, Scan-ARE, Greedy+Merge, BloomARE}
//
// SOSD Books is the headline distribution because (a) the FPR-vs-BPK plot
// uses Books as one of the headlines, (b) Books has small universe so it
// stresses Grafite's bpk envelope (an honest test rather than a wide-margin
// freebie), and (c) Books is uint32, which exercises the SuRF wrapper at a
// non-pathological key width.
func TestB6IndustryLatency(t *testing.T) {
	const (
		n          = 1 << 24
		queryCount = 1 << 18 // 256K queries — big enough to amortise CGo dispatch
		eps        = 0.01
	)
	rangeLens := []uint64{1, 16, 128, 1024, 4096, 16384, 65536}

	t.Logf("loading SOSD Books at n=%d", n)
	allKeys, err := loadSOSDUint32(sosdPath("books_200M_uint32"), 2*n)
	if err != nil {
		t.Skipf("SOSD Books unavailable: %v (run bench/sosd_data/download.sh)", err)
	}
	if len(allKeys) < n {
		t.Skipf("not enough unique Books keys: have %d, need %d", len(allKeys), n)
	}
	keys := allKeys[:n]
	keyBits := uint32(max(1, mathbits.Len64(keys[len(keys)-1])))
	t.Logf("loaded %d unique uint64 keys, range [%d, %d], keyBits=%d",
		len(keys), keys[0], keys[n-1], keyBits)

	type result struct {
		Filter        string  `json:"filter"`
		RangeLen      uint64  `json:"rangeLen"`
		BuildNs       int64   `json:"buildNs"`
		BuildMKeysSec float64 `json:"buildMKeysSec"`
		QueryNsPerOp  float64 `json:"queryNsPerOp"`
		BPKUsed       float64 `json:"bpkUsed"`
		Note          string  `json:"note,omitempty"`
	}
	var rows []result

	queryRng := rand.New(rand.NewSource(20260430))
	queries := generateRangeQueries(keys, queryCount, rangeLens[0], queryRng)
	_ = queries // ensure helper compiles; per-L queries built below

	type filterDef struct {
		name string
		// build returns an isEmpty closure (uint64,uint64)->bool, plus the bpk used.
		build func(L uint64) (isEmpty func(a, b uint64) bool, bpk float64, err error)
	}

	filters := []filterDef{
		{"SODA", func(L uint64) (func(a, b uint64) bool, float64, error) {
			f, err := are_soda_hash.NewSodaARE(keys, L, eps)
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, math.Log2(float64(L) / eps), nil
		}},
		{"Truncation", func(L uint64) (func(a, b uint64) bool, float64, error) {
			f, err := are_trunc.NewTruncARE(keys, keyBits, are_trunc.Config{Eps: eps})
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, math.Log2(float64(L) / eps), nil
		}},
		{"Scan-ARE", func(L uint64) (func(a, b uint64) bool, float64, error) {
			f, err := are_hybrid_scan.NewHybridScanARE(keys, keyBits,
				are_hybrid_scan.Config{RangeLen: float64(L), Eps: eps})
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, math.Log2(float64(L) / eps), nil
		}},
		{"Greedy+Merge", func(L uint64) (func(a, b uint64) bool, float64, error) {
			f, err := are_greedy_scan.NewGreedyScanARE(keys, keyBits,
				are_greedy_scan.Config{RangeLen: float64(L), Eps: eps})
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, math.Log2(float64(L) / eps), nil
		}},
		{"BloomARE", func(L uint64) (func(a, b uint64) bool, float64, error) {
			f, err := are_bloom.NewBloomARE(keys, L, eps)
			if err != nil {
				return nil, 0, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, math.Log2(float64(L) / eps), nil
		}},
		{"Grafite", func(L uint64) (func(a, b uint64) bool, float64, error) {
			bpk := math.Log2(float64(L) / eps)
			f := tryGrafite(keys, bpk)
			if f == nil {
				return nil, bpk, fmt.Errorf("grafite: bpk=%.2f exceeds envelope", bpk)
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, bpk, nil
		}},
		{"SNARF", func(L uint64) (func(a, b uint64) bool, float64, error) {
			bpk := math.Log2(float64(L) / eps)
			f := snarf.New(keys, bpk)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, bpk, nil
		}},
		{"SuRFReal(8)", func(L uint64) (func(a, b uint64) bool, float64, error) {
			f := surf.New(keys, surf.SuffixReal, 0, 8)
			// SuRF doesn't use bpk; report as -1 to signal not applicable.
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, -1, nil
		}},
	}

	fmt.Printf("\n=== B6: Build + Query latency on SOSD Books, n=2^24, ε=%.3f ===\n", eps)
	fmt.Printf("%-14s | %-7s | %-9s | %-13s | %-13s | %-7s\n",
		"Filter", "L", "build_ms", "build_Mkeys/s", "query_ns/op", "bpk")

	for _, fd := range filters {
		// Warm-up — small build + tiny query batch — flushes CGo lazy init.
		warmKeys := keys[:1<<10]
		warmIsEmpty, _, werr := fd.build(rangeLens[0])
		if werr == nil {
			_ = warmIsEmpty(warmKeys[0], warmKeys[len(warmKeys)-1])
		}
		runtime.GC()

		for _, L := range rangeLens {
			startBuild := time.Now()
			isEmpty, bpk, err := fd.build(L)
			buildDur := time.Since(startBuild)
			if err != nil {
				rows = append(rows, result{
					Filter:   fd.name,
					RangeLen: L,
					BPKUsed:  bpk,
					Note:     err.Error(),
				})
				fmt.Printf("%-14s | L=%-5d | %-9s | %-13s | %-13s | %-7.2f  %s\n",
					fd.name, L, "—", "—", "—", bpk, err.Error())
				continue
			}
			buildMKeys := float64(n) / buildDur.Seconds() / 1e6

			qrng := rand.New(rand.NewSource(int64(L) + 7777777))
			batch := generateRangeQueries(keys, queryCount, L, qrng)

			startQ := time.Now()
			for _, q := range batch {
				isEmpty(q[0], q[1])
			}
			qDur := time.Since(startQ)
			nsPerQuery := float64(qDur.Nanoseconds()) / float64(queryCount)

			rows = append(rows, result{
				Filter:        fd.name,
				RangeLen:      L,
				BuildNs:       buildDur.Nanoseconds(),
				BuildMKeysSec: buildMKeys,
				QueryNsPerOp:  nsPerQuery,
				BPKUsed:       bpk,
			})
			fmt.Printf("%-14s | L=%-5d | %-9.1f | %-13.2f | %-13.1f | %-7.2f\n",
				fd.name, L, float64(buildDur.Milliseconds()), buildMKeys, nsPerQuery, bpk)
		}
	}

	dataDir := "../bench_results/data"
	os.MkdirAll(dataDir, 0755)
	outPath := dataDir + "/b6_latency.json"
	doc := struct {
		Type         string   `json:"type"`
		Distribution string   `json:"distribution"`
		NKeys        int      `json:"nKeys"`
		QueryCount   int      `json:"queryCount"`
		Eps          float64  `json:"eps"`
		KeyBits      uint32   `json:"keyBits"`
		Timestamp    string   `json:"timestamp"`
		GitCommit    string   `json:"gitCommit"`
		Rows         []result `json:"rows"`
	}{
		Type:         "b6_latency",
		Distribution: "sosd_books",
		NKeys:        n,
		QueryCount:   queryCount,
		Eps:          eps,
		KeyBits:      keyBits,
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		GitCommit:    gitCommitShort(),
		Rows:         rows,
	}
	buf, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(outPath, buf, 0644); err != nil {
		t.Fatalf("write %s: %v", outPath, err)
	}
	t.Logf("wrote %s", outPath)
}
