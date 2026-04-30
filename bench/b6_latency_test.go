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
	"sync"
	"testing"
	"time"
)

// TestB6IndustryLatency measures build throughput and query latency for the
// headline filter set at n=2^24 across a few representative range lengths.
// Output for the `B6` defence backup slide and §sec:eval-build-query-latency.
//
// Layout: each filter is its own subtest (t.Run), so:
//   - a SIGSEGV in one CGo wrapper does not lose data collected by previous
//     filters (subtest results are flushed to JSON as soon as a subtest
//     finishes, not at the end of the whole test);
//   - individual filters can be re-run via `-run TestB6IndustryLatency/SODA`
//     and their rows merged into the existing JSON in place;
//   - parallelism per filter could be enabled later via t.Parallel() if
//     filter builds become expensive, though for now we keep it serial so
//     the timings are clean.
//
// Output:
//   - bench_results/data/b6_latency.json (machine-readable, incrementally
//     updated per subtest)
//
// Distribution choice: SOSD FB (sparse uint64 universe ~2^48). Books is too
// dense for an honest BloomARE measurement.
//
// Query mix: generateSmartQueries — 80% guaranteed-empty (in-gap +
// near-key truncated to gap), 20% uniform random. Empty queries force
// BloomARE to walk all L positions; ERE-based filters do their constant
// work regardless.
func TestB6IndustryLatency(t *testing.T) {
	const (
		n          = 1 << 24
		queryCount = 1 << 18
		eps        = 0.01
	)
	rangeLens := []uint64{1, 16, 128, 1024, 4096, 16384, 65536}

	t.Logf("loading SOSD FB at n=%d", n)
	allKeys, err := loadSOSDUint64(sosdPath("fb_200M_uint64"), 2*n)
	if err != nil {
		t.Skipf("SOSD FB unavailable: %v (run bench/sosd_data/download.sh)", err)
	}
	if len(allKeys) < n {
		t.Skipf("not enough unique FB keys: have %d, need %d", len(allKeys), n)
	}
	keys := allKeys[:n]
	keyBits := uint32(max(1, mathbits.Len64(keys[len(keys)-1])))
	t.Logf("loaded %d unique uint64 keys, range [%d, %d], keyBits=%d",
		len(keys), keys[0], keys[n-1], keyBits)

	type filterDef struct {
		name  string
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
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, -1, nil
		}},
	}

	fmt.Printf("\n=== B6: Build + Query latency on SOSD FB, n=2^24, ε=%.3f ===\n", eps)
	fmt.Printf("%-14s | %-7s | %-9s | %-13s | %-13s | %-7s\n",
		"Filter", "L", "build_ms", "build_Mkeys/s", "query_ns/op", "bpk")

	store := newB6Store(n, queryCount, eps, keyBits)

	for _, fd := range filters {
		fd := fd
		t.Run(fd.name, func(t *testing.T) {
			rows := runB6Filter(t, fd.name, fd.build, keys, rangeLens, queryCount, n)
			store.update(fd.name, rows)
			if err := store.flush(); err != nil {
				t.Errorf("flush b6_latency.json: %v", err)
			}
		})
	}

	t.Logf("wrote %s", store.path())
}

type b6Row struct {
	Filter        string  `json:"filter"`
	RangeLen      uint64  `json:"rangeLen"`
	BuildNs       int64   `json:"buildNs"`
	BuildMKeysSec float64 `json:"buildMKeysSec"`
	QueryNsPerOp  float64 `json:"queryNsPerOp"`
	BPKUsed       float64 `json:"bpkUsed"`
	Note          string  `json:"note,omitempty"`
}

type b6Doc struct {
	Type          string  `json:"type"`
	Distribution  string  `json:"distribution"`
	NKeys         int     `json:"nKeys"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	Eps           float64 `json:"eps"`
	KeyBits       uint32  `json:"keyBits"`
	Timestamp     string  `json:"timestamp"`
	GitCommit     string  `json:"gitCommit"`
	Rows          []b6Row `json:"rows"`
}

// b6Store is the per-filter incremental writer for b6_latency.json.
// It loads any pre-existing rows on first use so that re-running a single
// subtest (`-run TestB6IndustryLatency/Grafite`) merges into the prior
// run's output instead of clobbering it.
type b6Store struct {
	mu  sync.Mutex
	doc b6Doc
}

func newB6Store(nKeys, queryCount int, eps float64, keyBits uint32) *b6Store {
	s := &b6Store{
		doc: b6Doc{
			Type:          "b6_latency",
			Distribution:  "sosd_fb",
			NKeys:         nKeys,
			QueryCount:    queryCount,
			QueryStrategy: "smart_mix_guaranteed_empty",
			Eps:           eps,
			KeyBits:       keyBits,
		},
	}
	// Best-effort load of prior rows. Any field other than Rows is
	// re-derived from the current run's parameters; we only carry forward
	// rows for filters that we are not about to re-measure.
	if buf, err := os.ReadFile(s.path()); err == nil {
		var prior b6Doc
		if err := json.Unmarshal(buf, &prior); err == nil {
			s.doc.Rows = prior.Rows
		}
	}
	return s
}

func (s *b6Store) path() string {
	return "../bench_results/data/b6_latency.json"
}

// update replaces all rows for `filter` with the new ones (so re-running a
// subtest produces deterministic output, not appended duplicates).
func (s *b6Store) update(filter string, rows []b6Row) {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.doc.Rows[:0]
	for _, r := range s.doc.Rows {
		if r.Filter != filter {
			kept = append(kept, r)
		}
	}
	s.doc.Rows = append(kept, rows...)
}

func (s *b6Store) flush() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.doc.Timestamp = time.Now().UTC().Format(time.RFC3339)
	s.doc.GitCommit = gitCommitShort()
	if err := os.MkdirAll("../bench_results/data", 0755); err != nil {
		return err
	}
	buf, err := json.MarshalIndent(s.doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path(), buf, 0644)
}

func runB6Filter(
	t *testing.T,
	name string,
	build func(L uint64) (func(a, b uint64) bool, float64, error),
	keys []uint64,
	rangeLens []uint64,
	queryCount int,
	n int,
) []b6Row {
	// Warm-up at smallest L flushes CGo lazy init / page faults / TLB.
	warmKeys := keys[:1<<10]
	if warmIsEmpty, _, werr := build(rangeLens[0]); werr == nil {
		_ = warmIsEmpty(warmKeys[0], warmKeys[len(warmKeys)-1])
	}
	runtime.GC()

	rows := make([]b6Row, 0, len(rangeLens))
	for _, L := range rangeLens {
		startBuild := time.Now()
		isEmpty, bpk, err := build(L)
		buildDur := time.Since(startBuild)
		if err != nil {
			rows = append(rows, b6Row{
				Filter:   name,
				RangeLen: L,
				BPKUsed:  bpk,
				Note:     err.Error(),
			})
			fmt.Printf("%-14s | L=%-5d | %-9s | %-13s | %-13s | %-7.2f  %s\n",
				name, L, "—", "—", "—", bpk, err.Error())
			continue
		}
		buildMKeys := float64(n) / buildDur.Seconds() / 1e6

		qrng := rand.New(rand.NewSource(int64(L) + 7777777))
		batch := generateSmartQueries(keys, queryCount, L, qrng)
		if len(batch) == 0 {
			t.Logf("%s L=%d: smart-query generator returned 0 queries; skipping", name, L)
			continue
		}

		startQ := time.Now()
		for _, q := range batch {
			isEmpty(q[0], q[1])
		}
		qDur := time.Since(startQ)
		nsPerQuery := float64(qDur.Nanoseconds()) / float64(len(batch))

		rows = append(rows, b6Row{
			Filter:        name,
			RangeLen:      L,
			BuildNs:       buildDur.Nanoseconds(),
			BuildMKeysSec: buildMKeys,
			QueryNsPerOp:  nsPerQuery,
			BPKUsed:       bpk,
		})
		fmt.Printf("%-14s | L=%-5d | %-9.1f | %-13.2f | %-13.1f | %-7.2f\n",
			name, L, float64(buildDur.Milliseconds()), buildMKeys, nsPerQuery, bpk)
	}
	return rows
}
