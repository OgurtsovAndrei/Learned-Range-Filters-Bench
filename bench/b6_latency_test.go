package bench_test

import (
	"Thesis-bench-industry/thirdparty/snarf"
	"Thesis-bench-industry/thirdparty/surf"
	"Thesis/emptiness/approx/are_bloom"
	"Thesis/emptiness/approx/are_greedy_scan"
	"Thesis/emptiness/approx/are_hybrid_scan"
	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/emptiness/approx/are_trunc"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	mathbits "math/bits"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestB6IndustryLatency measures four things per (distribution, filter, L):
//   - build throughput (M keys/sec)
//   - actual filter footprint (BPK = SizeInBits / n)
//   - query latency (ns/op) on guaranteed-empty smart-mix queries
//   - empirical FPR (= count of false-positive returns / queries),
//     since smart-mix queries are guaranteed empty
//
// One pass over the query batch covers both latency and FPR — no second
// query batch needed. Headline filter set at n=2^24 across L values, on
// multiple distributions.
//
// Layout:
//
//	TestB6IndustryLatency/<distribution>/<filter>
//
// Each (distribution, filter) is its own subtest so a SIGSEGV in one CGo
// wrapper does not lose data collected by previous subtests, and one filter
// can be re-run via:
//
//	go test -run "TestB6IndustryLatency/sosd_fb/Scan-ARE" ./bench/
//
// Distribution diversity is important here: SODA has a known degenerate
// regime when the input universe fits in a single super-block of size 2^K
// (K = log2(n*L/eps)). On SOSD FB (keys < 2^33) and L >= 1, K already
// exceeds the key width, so PairwiseHash(0, ...) = 0 and SODA's hashing
// reduces to identity. The ERE underneath then sees the original
// non-uniform key distribution. Synthetic uniform / spread (keys spread
// over 2^64) and SOSD osm (uint64 cell IDs in a wider universe) test the
// non-degenerate regime where SODA actually mixes keys across blocks.
//
// Output: bench_results/data/b6_latency.json (incrementally updated per
// subtest, merged across runs).
func TestB6IndustryLatency(t *testing.T) {
	const (
		n          = 1 << 24
		queryCount = 1 << 18
		eps        = 0.01
	)
	rangeLens := []uint64{1, 16, 128, 1024, 4096, 16384, 65536}

	type distSpec struct {
		name string
		// load returns at least n unique uint64 keys (sorted).
		load func() ([]uint64, error)
	}
	distributions := []distSpec{
		{"sosd_fb", func() ([]uint64, error) {
			return loadSOSDUint64(sosdPath("fb_200M_uint64"), 2*n)
		}},
		{"sosd_wiki", func() ([]uint64, error) {
			return loadSOSDUint64(sosdPath("wiki_ts_200M_uint64"), 2*n)
		}},
		{"sosd_osm", func() ([]uint64, error) {
			return loadSOSDUint64(sosdPath("osm_cellids_800M_uint64"), 2*n)
		}},
		{"sosd_books", func() ([]uint64, error) {
			return loadSOSDUint32(sosdPath("books_200M_uint32"), 2*n)
		}},
		{"uniform", func() ([]uint64, error) {
			return loadSOSDUint64(syntheticDataPath("uniform_16M_uint64"), 0)
		}},
		{"spread", func() ([]uint64, error) {
			return loadSOSDUint64(syntheticDataPath("spread_16M_uint64"), 0)
		}},
		{"clustered", func() ([]uint64, error) {
			return loadSOSDUint64(syntheticDataPath("clustered_16M_uint64"), 0)
		}},
	}

	skipSet := map[string]bool{}
	if skip := os.Getenv("SKIP_FILTERS"); skip != "" {
		for _, name := range strings.Split(skip, ",") {
			skipSet[strings.TrimSpace(name)] = true
		}
	}

	store := newB6Store(n, queryCount, eps)
	fmt.Printf("\n=== B6: Build + Query latency + actual BPK + FPR, n=2^24, ε=%.3f ===\n", eps)
	fmt.Printf("%-11s | %-14s | %-7s | %-9s | %-13s | %-13s | %-7s | %-9s\n",
		"Distribution", "Filter", "L", "build_ms", "build_Mkeys/s", "query_ns/op", "bpk", "fpr")

	for _, ds := range distributions {
		ds := ds
		t.Run(ds.name, func(t *testing.T) {
			allKeys, err := ds.load()
			if err != nil {
				t.Skipf("load %s: %v", ds.name, err)
			}
			if len(allKeys) < n {
				t.Skipf("not enough keys for %s: have %d, need %d", ds.name, len(allKeys), n)
			}
			keys := allKeys[:n]
			keyBits := uint32(max(1, mathbits.Len64(keys[len(keys)-1])))
			t.Logf("%s: %d keys, range [%d, %d], keyBits=%d",
				ds.name, len(keys), keys[0], keys[n-1], keyBits)

			filters := buildB6Filters(keys, keyBits, eps)
			for _, fd := range filters {
				fd := fd
				if skipSet[fd.name] {
					t.Logf("%s/%s: skipped via SKIP_FILTERS", ds.name, fd.name)
					continue
				}
				t.Run(fd.name, func(t *testing.T) {
					rows := runB6Filter(t, store, ds.name, fd.name, fd.build,
						keys, rangeLens, queryCount, n, eps)
					store.update(ds.name, fd.name, rows)
					if err := store.flush(); err != nil {
						t.Errorf("flush b6_latency.json: %v", err)
					}
				})
			}
		})
	}

	t.Logf("wrote %s", store.path())
}

type b6FilterDef struct {
	name string
	// build returns isEmpty closure plus the actual filter footprint (bits).
	// sizeBits is then divided by n by the runner to get actual BPK.
	build func(L uint64) (isEmpty func(a, b uint64) bool, sizeBits uint64, err error)
}

func buildB6Filters(keys []uint64, keyBits uint32, eps float64) []b6FilterDef {
	return []b6FilterDef{
		{"SODA", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			f, err := are_soda_hash.NewSodaARE(keys, L, eps)
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, f.SizeInBits(), nil
		}},
		{"Truncation", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			f, err := are_trunc.NewTruncARE(keys, keyBits, are_trunc.Config{Eps: eps})
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, f.SizeInBits(), nil
		}},
		{"Scan-ARE", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			f, err := are_hybrid_scan.NewHybridScanARE(keys, keyBits,
				are_hybrid_scan.Config{RangeLen: float64(L), Eps: eps})
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, f.SizeInBits(), nil
		}},
		{"Greedy+Merge", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			f, err := are_greedy_scan.NewGreedyScanARE(keys, keyBits,
				are_greedy_scan.Config{RangeLen: float64(L), Eps: eps})
			if err != nil {
				return nil, 0, err
			}
			return f.IsEmpty, f.SizeInBits(), nil
		}},
		{"BloomARE", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			f, err := are_bloom.NewBloomARE(keys, L, eps)
			if err != nil {
				return nil, 0, err
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
		}},
		{"Grafite", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			bpk := math.Log2(float64(L) / eps)
			f := tryGrafite(keys, bpk)
			if f == nil {
				return nil, 0, fmt.Errorf("grafite: target bpk=%.2f exceeds envelope", bpk)
			}
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
		}},
		{"SNARF", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			bpk := math.Log2(float64(L) / eps)
			f := snarf.New(keys, bpk)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
		}},
		{"SuRFReal(8)", func(L uint64) (func(a, b uint64) bool, uint64, error) {
			f := surf.New(keys, surf.SuffixReal, 0, 8)
			return func(a, b uint64) bool { return f.IsEmpty(a, b) }, f.SizeInBits(), nil
		}},
	}
}

type b6Row struct {
	Distribution  string  `json:"distribution"`
	Filter        string  `json:"filter"`
	RangeLen      uint64  `json:"rangeLen"`
	BuildNs       int64   `json:"buildNs"`
	BuildMKeysSec float64 `json:"buildMKeysSec"`
	QueryNsPerOp  float64 `json:"queryNsPerOp"`
	BPKUsed       float64 `json:"bpkUsed"`
	SizeBits      uint64  `json:"sizeBits"`
	FPR           float64 `json:"fpr"`
	QueriesEmpty  int     `json:"queriesEmpty"`
	ParamsHash    string  `json:"paramsHash"`
	Note          string  `json:"note,omitempty"`
}

// b6Params captures the hyperparameters that, if changed, invalidate a
// cached row. Stored hashed into b6Row.ParamsHash; the runner computes the
// hash for the current run and skips rows whose stored hash matches.
type b6Params struct {
	NKeys         int     `json:"nKeys"`
	Eps           float64 `json:"eps"`
	RangeLen      uint64  `json:"rangeLen"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	QuerySeed     int64   `json:"querySeed"`
	Distribution  string  `json:"distribution"`
	Filter        string  `json:"filter"`
}

func (p b6Params) hash() string {
	buf, _ := json.Marshal(p)
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:8]) // 16 hex chars is plenty for collisions
}

type b6Doc struct {
	Type          string  `json:"type"`
	NKeys         int     `json:"nKeys"`
	QueryCount    int     `json:"queryCount"`
	QueryStrategy string  `json:"queryStrategy"`
	Eps           float64 `json:"eps"`
	Timestamp     string  `json:"timestamp"`
	GitCommit     string  `json:"gitCommit"`
	Rows          []b6Row `json:"rows"`
}

// b6Store is the per-(distribution,filter) incremental writer. On first use
// it loads any prior b6_latency.json so re-running a single subtest merges
// into the prior run's output.
type b6Store struct {
	mu  sync.Mutex
	doc b6Doc
}

func newB6Store(nKeys, queryCount int, eps float64) *b6Store {
	s := &b6Store{
		doc: b6Doc{
			Type:          "b6_latency",
			NKeys:         nKeys,
			QueryCount:    queryCount,
			QueryStrategy: "smart_mix_guaranteed_empty",
			Eps:           eps,
		},
	}
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

func (s *b6Store) update(dist, filter string, rows []b6Row) {
	s.mu.Lock()
	defer s.mu.Unlock()
	kept := s.doc.Rows[:0]
	for _, r := range s.doc.Rows {
		if !(r.Distribution == dist && r.Filter == filter) {
			kept = append(kept, r)
		}
	}
	s.doc.Rows = append(kept, rows...)
}

// cachedRow returns a prior row for (dist, filter, L) whose paramsHash
// matches the requested one, or nil. Use to short-circuit measurement when
// FORCE is unset.
func (s *b6Store) cachedRow(dist, filter string, L uint64, paramsHash string) *b6Row {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.doc.Rows {
		r := &s.doc.Rows[i]
		if r.Distribution == dist && r.Filter == filter &&
			r.RangeLen == L && r.ParamsHash == paramsHash {
			return r
		}
	}
	return nil
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
	store *b6Store,
	dist, name string,
	build func(L uint64) (func(a, b uint64) bool, uint64, error),
	keys []uint64,
	rangeLens []uint64,
	queryCount int,
	n int,
	eps float64,
) []b6Row {
	force := os.Getenv("FORCE") != ""

	rows := make([]b6Row, 0, len(rangeLens))
	warmedUp := false
	for _, L := range rangeLens {
		params := b6Params{
			NKeys:         n,
			Eps:           eps,
			RangeLen:      L,
			QueryCount:    queryCount,
			QueryStrategy: "smart_mix_guaranteed_empty",
			QuerySeed:     int64(L) + 7777777,
			Distribution:  dist,
			Filter:        name,
		}
		paramsHash := params.hash()

		if !force {
			if cached := store.cachedRow(dist, name, L, paramsHash); cached != nil {
				rows = append(rows, *cached)
				fmt.Printf("%-11s | %-14s | L=%-5d | %-9s | %-13s | %-13s | %-7s | %-9s  (cached)\n",
					dist, name, L, "—", "—", "—", "—", "—")
				continue
			}
		}

		// Lazy warm-up — only flush CGo init / page faults once we know we'll
		// actually measure something this iteration.
		if !warmedUp {
			warmKeys := keys[:1<<10]
			if warmIsEmpty, _, werr := build(rangeLens[0]); werr == nil {
				_ = warmIsEmpty(warmKeys[0], warmKeys[len(warmKeys)-1])
			}
			runtime.GC()
			warmedUp = true
		}

		startBuild := time.Now()
		isEmpty, sizeBits, err := build(L)
		buildDur := time.Since(startBuild)
		if err != nil {
			rows = append(rows, b6Row{
				Distribution: dist,
				Filter:       name,
				RangeLen:     L,
				ParamsHash:   paramsHash,
				Note:         err.Error(),
			})
			fmt.Printf("%-11s | %-14s | L=%-5d | %-9s | %-13s | %-13s | %-7s | %-9s  %s\n",
				dist, name, L, "—", "—", "—", "—", "—", err.Error())
			continue
		}
		buildMKeys := float64(n) / buildDur.Seconds() / 1e6
		actualBPK := float64(sizeBits) / float64(n)

		qrng := rand.New(rand.NewSource(params.QuerySeed))
		batch := generateSmartQueries(keys, queryCount, L, qrng)
		if len(batch) == 0 {
			t.Logf("%s/%s L=%d: smart-query generator returned 0 queries; skipping",
				dist, name, L)
			continue
		}

		// Smart queries are guaranteed empty, so any false return is a
		// false positive. We count FPs in the same loop as the latency
		// measurement — no extra query work, and no separate batch.
		falsePositives := 0
		startQ := time.Now()
		for _, q := range batch {
			if !isEmpty(q[0], q[1]) {
				falsePositives++
			}
		}
		qDur := time.Since(startQ)
		nsPerQuery := float64(qDur.Nanoseconds()) / float64(len(batch))
		fpr := float64(falsePositives) / float64(len(batch))

		rows = append(rows, b6Row{
			Distribution:  dist,
			Filter:        name,
			RangeLen:      L,
			BuildNs:       buildDur.Nanoseconds(),
			BuildMKeysSec: buildMKeys,
			QueryNsPerOp:  nsPerQuery,
			BPKUsed:       actualBPK,
			SizeBits:      sizeBits,
			FPR:           fpr,
			QueriesEmpty:  len(batch),
			ParamsHash:    paramsHash,
		})
		fmt.Printf("%-11s | %-14s | L=%-5d | %-9.1f | %-13.2f | %-13.1f | %-7.2f | %-9.4f\n",
			dist, name, L, float64(buildDur.Milliseconds()), buildMKeys, nsPerQuery, actualBPK, fpr)
	}
	return rows
}
