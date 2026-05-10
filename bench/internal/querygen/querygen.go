package querygen

import (
	"math/rand"
	"sort"
)

// SmartMixWeights captures the per-bucket fractions for GenerateSmartQueriesWeighted.
// Fields must sum to 1.0.
type SmartMixWeights struct {
	NearKey float64
	InGap   float64
	Uniform float64
}

// Default smart-mix weights.
const (
	QueryWeightNearKey = 0.50 // query offset from a random key
	QueryWeightInGap   = 0.30 // query placed inside a random gap
	QueryWeightUniform = 0.20 // uniform random across span

	// Mask60 is used for key masking.
	Mask60 = (uint64(1) << 60) - 1
)

var DefaultSmartMix = SmartMixWeights{
	NearKey: QueryWeightNearKey,
	InGap:   QueryWeightInGap,
	Uniform: QueryWeightUniform,
}

// GenerateSmartQueries generates a mix of query types that follow the data
// distribution. Every returned query is guaranteed to be empty.
func GenerateSmartQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	return GenerateSmartQueriesWeighted(keys, count, rangeLen, DefaultSmartMix, rng)
}

// GenerateRangeQueries generates uniform queries. If keys is provided,
// it uses the [minK, maxK] range; otherwise it uses full 60-bit range.
func GenerateRangeQueries(keys []uint64, count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	var minK, maxK uint64
	var span uint64
	if len(keys) >= 2 {
		minK, maxK = keys[0], keys[len(keys)-1]
		span = maxK - minK
	} else {
		minK = 0
		maxK = Mask60
		span = Mask60
	}
	
	queries := make([][2]uint64, count)
	for i := range queries {
		a := minK + randUint64Below(rng, span)
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

func GenerateZipfianQueries(count int, prefixes []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	nTop := len(prefixes) / 10
	if nTop == 0 {
		return nil
	}
	queries := make([][2]uint64, count)
	nHotQ := count * 80 / 100
	for i := 0; i < nHotQ; i++ {
		pref := prefixes[rng.Intn(nTop)]
		a := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		a &= Mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	for i := nHotQ; i < count; i++ {
		a := rng.Uint64() & Mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

func GenerateTemporalQueries(count int, keys []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	if len(keys) < 2 {
		return nil
	}
	queries := make([][2]uint64, count)
	minK, maxK := keys[0], keys[len(keys)-1]
	spread := maxK - minK
	for i := range queries {
		var a uint64
		if rng.Float64() < 0.5 {
			recentBase := maxK - spread*30/100
			a = recentBase + uint64(rng.Int63n(int64(spread*30/100)))
		} else {
			a = minK + uint64(rng.Int63n(int64(spread)))
		}
		a &= Mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

// randUint64Below returns a uniformly random uint64 in [0, n).
func randUint64Below(rng *rand.Rand, n uint64) uint64 {
	if n == 0 {
		return 0
	}
	if n <= 1<<63 {
		return uint64(rng.Int63n(int64(n)))
	}
	return rng.Uint64() % n
}

// GenerateSmartQueriesWeighted is the parametrized variant.
func GenerateSmartQueriesWeighted(keys []uint64, count int, rangeLen uint64, w SmartMixWeights, rng *rand.Rand) [][2]uint64 {
	n := len(keys)
	if n < 2 {
		return nil
	}
	minK, maxK := keys[0], keys[n-1]
	span := maxK - minK
	if span == 0 {
		return nil
	}

	nNear := int(float64(count) * w.NearKey)
	nGap := int(float64(count) * w.InGap)
	nUnif := count - nNear - nGap

	// Pre-compute gaps for gap-sampling.
	type gap struct{ lo, hi uint64 }
	gaps := make([]gap, 0, n-1)
	for i := 0; i < n-1; i++ {
		if keys[i+1]-keys[i] > 1 {
			gaps = append(gaps, gap{keys[i] + 1, keys[i+1] - 1})
		}
	}

	queries := make([][2]uint64, 0, count)

	tryAdd := func(a, b uint64) {
		if b < a || a == 0 && b == 0 {
			return
		}
		// Clamp to key range.
		if a < minK && minK > rangeLen {
			a = minK - rangeLen
		}
		if b > maxK+rangeLen {
			b = maxK + rangeLen
		}
		// Find first key >= a.
		idx := sort.Search(n, func(i int) bool { return keys[i] >= a })
		if idx < n && keys[idx] <= b {
			// Query contains key[idx] — truncate to [a, key[idx]-1].
			if keys[idx] == 0 || keys[idx]-1 < a {
				return // can't truncate, skip
			}
			b = keys[idx] - 1
		}
		if b >= a {
			queries = append(queries, [2]uint64{a, b})
		}
	}

	// Near-key queries: pick a random key, offset by [-5*rangeLen, +5*rangeLen].
	for i := 0; i < nNear*2 && len(queries) < nNear; i++ {
		key := keys[rng.Intn(n)]
		offset := rng.Int63n(int64(rangeLen) * 10)
		offset -= int64(rangeLen) * 5
		a := int64(key) + offset
		if a < 0 {
			a = 0
		}
		tryAdd(uint64(a), uint64(a)+rangeLen-1)
	}

	// In-gap queries: pick a random gap, place query inside.
	target := nNear + nGap
	if len(gaps) > 0 {
		for i := 0; i < nGap*2 && len(queries) < target; i++ {
			g := gaps[rng.Intn(len(gaps))]
			gapLen := g.hi - g.lo + 1
			if gapLen == 0 {
				continue
			}
			a := g.lo + randUint64Below(rng, gapLen)
			b := a + rangeLen - 1
			if b > g.hi {
				b = g.hi
			}
			if b >= a {
				queries = append(queries, [2]uint64{a, b}) // guaranteed empty (inside gap)
			}
		}
	}

	// Uniform queries: random across span.
	target = count
	for i := 0; i < nUnif*2 && len(queries) < target; i++ {
		a := minK + randUint64Below(rng, span)
		tryAdd(a, a+rangeLen-1)
	}

	return queries
}
