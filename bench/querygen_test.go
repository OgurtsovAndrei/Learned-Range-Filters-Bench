package bench_test

import (
	"math/rand"
)

func generateUniformQueries(count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	queries := make([][2]uint64, count)
	for i := range queries {
		a := rng.Uint64() & mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

func generateZipfianQueries(count int, prefixes []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	nTop := len(prefixes) / 10
	queries := make([][2]uint64, count)
	nHotQ := count * 80 / 100
	for i := 0; i < nHotQ; i++ {
		pref := prefixes[rng.Intn(nTop)]
		a := (pref << 20) | (rng.Uint64() & ((1 << 20) - 1))
		a &= mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	for i := nHotQ; i < count; i++ {
		a := rng.Uint64() & mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}

func generateTemporalQueries(count int, keys []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
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
		a &= mask60
		queries[i] = [2]uint64{a, a + rangeLen - 1}
	}
	return queries
}
