package bench_test

import (
	"Thesis-bench-industry/bench/internal/querygen"
	"math/rand"
)


func generateUniformQueries(count int, rangeLen uint64, rng *rand.Rand) [][2]uint64 {

	return querygen.GenerateRangeQueries(nil, count, rangeLen, rng)
}

func generateZipfianQueries(count int, prefixes []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	return querygen.GenerateZipfianQueries(count, prefixes, rangeLen, rng)
}

func generateTemporalQueries(count int, keys []uint64, rangeLen uint64, rng *rand.Rand) [][2]uint64 {
	return querygen.GenerateTemporalQueries(count, keys, rangeLen, rng)
}


