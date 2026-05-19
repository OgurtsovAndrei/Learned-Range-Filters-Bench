// nonempty_frac generates the smart-mix-mixed query workload for each
// (dataset, L) pair at a requested n and reports the fraction of queries
// that contain at least one stored key. Runs the same query generator as
// the b6 runner but skips the filter, so this is the cheapest way to
// extract the workload statistic without burning a benchmark slot.
//
// Usage:
//
//	go run ./bench/cmd/nonempty_frac -n 268435456
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"sort"

	"Thesis-bench-industry/bench/internal/keygen"
	"Thesis-bench-industry/bench/internal/querygen"
)

type distSpec struct {
	name    string
	loader  func(n int) ([]uint64, error)
	colName string
}

func main() {
	var (
		n          = flag.Int("n", 268435456, "key count")
		queryCount = flag.Int("q", 1<<18, "queries per (dataset, L)")
		seedBase   = flag.Int64("seed", 7777777, "base RNG seed (per-L seed = base + L)")
	)
	flag.Parse()

	dists := []distSpec{
		{"sosd_fb", func(n int) ([]uint64, error) { return keygen.LoadSOSDUint64(keygen.SOSDPath("fb_200M_uint64"), n) }, "Facebook"},
		{"sosd_wiki", func(n int) ([]uint64, error) { return keygen.LoadSOSDUint64(keygen.SOSDPath("wiki_ts_200M_uint64"), n) }, "Wiki"},
		{"sosd_books", func(n int) ([]uint64, error) { return keygen.LoadSOSDUint64(keygen.SOSDPath("books_800M_uint64"), n) }, "Books"},
		{"sosd_osm", func(n int) ([]uint64, error) { return keygen.LoadSOSDUint64(keygen.SOSDPath("osm_cellids_800M_uint64"), n) }, "OSM"},
		{"uniform", func(n int) ([]uint64, error) {
			return keygen.LoadSOSDUint64(fmt.Sprintf("bench/synthetic_data/uniform_%s_uint64", syntheticSize(n)), 0)
		}, "Uniform"},
		{"clustered", func(n int) ([]uint64, error) {
			return keygen.LoadSOSDUint64(fmt.Sprintf("bench/synthetic_data/clustered_%s_uint64", syntheticSize(n)), 0)
		}, "Clustered"},
	}

	Ls := []uint64{1, 16, 1024}
	weights := querygen.DefaultSmartMix

	fmt.Printf("%-10s | %-6s | %-9s | %-9s\n", "Dataset", "L", "queries", "non-empty")
	fmt.Println("---------------------------------------------------")

	for _, d := range dists {
		keys, err := d.loader(*n)
		if err != nil {
			fmt.Printf("%-10s : load failed: %v\n", d.colName, err)
			continue
		}
		if len(keys) > *n {
			keys = keys[:*n]
		}
		for _, L := range Ls {
			rng := rand.New(rand.NewSource(int64(L) + *seedBase))
			batch := querygen.GenerateMixedQueriesWeighted(keys, *queryCount, L, weights, rng)
			ne := 0
			for _, q := range batch {
				idx := sort.Search(len(keys), func(i int) bool { return keys[i] >= q[0] })
				if idx < len(keys) && keys[idx] <= q[1] {
					ne++
				}
			}
			frac := float64(ne) / float64(len(batch)) * 100
			fmt.Printf("%-10s | L=%-4d | %-9d | %.1f%%\n", d.colName, L, len(batch), frac)
		}
	}
}

// syntheticSize picks the smallest available synthetic-key file that
// can serve a request for n keys, mirroring runner_test.go::syntheticFile.
func syntheticSize(n int) string {
	if n > (1 << 24) {
		return "256M"
	}
	return "16M"
}
