package bench_test

import (
	"Thesis-bench-industry/grafite"
	"Thesis-bench-industry/snarf"
	"Thesis-bench-industry/surf"
	"math/rand"
	"sort"
	"testing"
)

func TestBatchConsistency(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	seen := make(map[uint64]bool)
	raw := make([]uint64, 0, 1000)
	for len(raw) < 1000 {
		v := rng.Uint64() & mask60
		if !seen[v] {
			seen[v] = true
			raw = append(raw, v)
		}
	}
	sort.Slice(raw, func(i, j int) bool { return raw[i] < raw[j] })

	queries := make([][2]uint64, 2048)
	for i := range queries {
		lo := rng.Uint64() & mask60
		hi := lo + uint64(rng.Intn(1000))
		if hi > mask60 {
			hi = mask60
		}
		queries[i] = [2]uint64{lo, hi}
	}

	type filterCase struct {
		name       string
		scalar     func(lo, hi uint64) bool
		batchQuery func([][2]uint64) []bool
	}

	gf := grafite.New(raw, 10.0)
	sf := snarf.New(raw, 10.0)
	surfBase := surf.New(raw, surf.SuffixNone, 0, 0)
	surfHash := surf.New(raw, surf.SuffixHash, 8, 0)
	surfReal := surf.New(raw, surf.SuffixReal, 0, 8)

	cases := []filterCase{
		{"Grafite", gf.IsEmpty, gf.QueryBatch},
		{"SNARF", sf.IsEmpty, sf.QueryBatch},
		{"SuRF", surfBase.IsEmpty, surfBase.QueryBatch},
		{"SuRFHash(8)", surfHash.IsEmpty, surfHash.QueryBatch},
		{"SuRFReal(8)", surfReal.IsEmpty, surfReal.QueryBatch},
	}

	for _, fc := range cases {
		t.Run(fc.name, func(t *testing.T) {
			batchResults := fc.batchQuery(queries)
			if len(batchResults) != len(queries) {
				t.Fatalf("batch returned %d results, expected %d", len(batchResults), len(queries))
			}
			mismatches := 0
			for i, q := range queries {
				scalar := fc.scalar(q[0], q[1])
				if scalar != batchResults[i] {
					mismatches++
					if mismatches <= 5 {
						t.Errorf("query %d: scalar=%v batch=%v (lo=%d hi=%d)", i, scalar, batchResults[i], q[0], q[1])
					}
				}
			}
			if mismatches > 5 {
				t.Errorf("... and %d more mismatches", mismatches-5)
			}
			if mismatches == 0 {
				t.Logf("%s: all %d queries match", fc.name, len(queries))
			}
		})
	}
}
