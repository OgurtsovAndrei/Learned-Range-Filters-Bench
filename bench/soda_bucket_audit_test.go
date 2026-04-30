package bench_test

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"Thesis/emptiness/approx/are_soda_hash"
)

// TestSodaBucketAudit dumps the actual bucket-size distribution of the
// inner ERE for SODA on real key sets, plus an empirical breakdown of
// (same-block vs multi-block) for smart-mix queries.
//
//	go test -run TestSodaBucketAudit -timeout 30m -v ./bench/
func TestSodaBucketAudit(t *testing.T) {
	const (
		n   = 1 << 24
		eps = 0.01
	)
	type spec struct {
		distName string
		L        uint64
	}
	specs := []spec{
		{"sosd_fb", 1},
		{"sosd_fb", 1024},
		{"sosd_fb", 65536},
		{"uniform", 65536},
	}
	for _, s := range specs {
		s := s
		t.Run(fmt.Sprintf("%s/L=%d", s.distName, s.L), func(t *testing.T) {
			keys := loadKeysForSpec(t, s.distName, n)
			t.Logf("min key %d, max key %d, span %d (~2^%.1f)",
				keys[0], keys[len(keys)-1],
				keys[len(keys)-1]-keys[0],
				logf(float64(keys[len(keys)-1]-keys[0])))
			ks := append([]uint64(nil), keys...)
			soda, err := are_soda_hash.NewSodaARE(ks, s.L, eps)
			if err != nil {
				t.Fatalf("NewSodaARE: %v", err)
			}
			sizes := soda.ERENonEmptyBlockSizes()
			if len(sizes) == 0 {
				t.Fatal("no populated blocks")
			}
			var sum, max int
			for _, sz := range sizes {
				sum += sz
				if sz > max {
					max = sz
				}
			}
			sortedSizes := append([]int(nil), sizes...)
			sort.Ints(sortedSizes)
			pct := func(p float64) int {
				idx := int(float64(len(sortedSizes)-1) * p)
				return sortedSizes[idx]
			}
			t.Logf("populated blocks: %d", len(sizes))
			t.Logf("avg bucket: %.1f, max bucket: %d", float64(sum)/float64(len(sizes)), max)
			t.Logf("p50=%d p90=%d p99=%d p999=%d",
				pct(0.50), pct(0.90), pct(0.99), pct(0.999))

			// Sample 1000 smart-mix-empty queries to see how many fall into
			// the same-block (1× ERE call) vs multi-block (2-3× ERE call) path.
			rng := rand.New(rand.NewSource(42))
			queries := generateSmartQueriesAudit(rng, keys, s.L, 1000)
			same, multi := 0, 0
			K := computeKAudit(n, s.L, eps)
			for _, q := range queries {
				blockA := q.lo >> K
				blockB := q.hi >> K
				if blockA == blockB {
					same++
				} else {
					multi++
				}
			}
			t.Logf("queries: same-block=%d, multi-block=%d", same, multi)
		})
	}
}

type queryAudit struct{ lo, hi uint64 }

// generateSmartQueriesAudit reuses the same "smart-mix empty" recipe the
// B6 bench uses: 80% near-key + 20% uniform, all guaranteed-empty.
func generateSmartQueriesAudit(rng *rand.Rand, keys []uint64, L uint64, count int) []queryAudit {
	out := make([]queryAudit, count)
	for i := range out {
		var lo uint64
		if rng.Intn(5) > 0 {
			k := keys[rng.Intn(len(keys))]
			off := uint64(rng.Intn(int(L*4))) + L
			if rng.Intn(2) == 0 {
				lo = k + off
			} else {
				if k > L+off {
					lo = k - off - L
				} else {
					lo = k + off
				}
			}
		} else {
			lo = rng.Uint64()
		}
		out[i] = queryAudit{lo: lo, hi: lo + L - 1}
	}
	return out
}

// computeKAudit replicates SODA's K = ceil(log2(n*L/eps)).
func computeKAudit(n int, L uint64, eps float64) uint32 {
	x := float64(n) * float64(L) / eps
	K := uint32(0)
	v := uint64(1)
	for v < uint64(x) {
		v <<= 1
		K++
	}
	return K
}

func logf(x float64) float64 {
	if x <= 0 {
		return 0
	}
	r := 0.0
	for x > 1 {
		x /= 2
		r++
	}
	return r
}
