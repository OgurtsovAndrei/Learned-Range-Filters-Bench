package bench_test

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestTradeoff_SOSD_Books(t *testing.T) {

	const queryCount = 1 << 18
	path := sosdPath("books_200M_uint32")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint32(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from books_200M, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName:   "sosd_books",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
				keySource: "sosd",
				keyFile:   "books_200M_uint32",
				queryStrategy: "smart_mix",
				queryStrategyParams: map[string]interface{}{
					"nearKeyWeight": queryWeightNearKey,
					"inGapWeight":   queryWeightInGap,
					"uniformWeight": queryWeightUniform,
				},
			})
		})
	}
}

func TestTradeoff_SOSD_Facebook(t *testing.T) {
	const queryCount = 1 << 20
	path := sosdPath("fb_200M_uint64")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint64(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from fb_200M, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName:   "sosd_fb",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
				keySource: "sosd",
				keyFile:   "fb_200M_uint64",
				queryStrategy: "smart_mix",
				queryStrategyParams: map[string]interface{}{
					"nearKeyWeight": queryWeightNearKey,
					"inGapWeight":   queryWeightInGap,
					"uniformWeight": queryWeightUniform,
				},
			})
		})
	}
}

func TestTradeoff_SOSD_Wiki(t *testing.T) {
	const queryCount = 1 << 18
	path := sosdPath("wiki_ts_200M_uint64")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint64(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from wiki_ts, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName:   "sosd_wiki",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
				keySource: "sosd",
				keyFile:   "wiki_ts_200M_uint64",
				queryStrategy: "smart_mix",
				queryStrategyParams: map[string]interface{}{
					"nearKeyWeight": queryWeightNearKey,
					"inGapWeight":   queryWeightInGap,
					"uniformWeight": queryWeightUniform,
				},
			})
		})
	}
}

func TestTradeoff_SOSD_OSM(t *testing.T) {
	const queryCount = 1 << 18
	path := sosdPath("osm_cellids_800M_uint64")

	for _, n := range []int{1 << 16, 1 << 18, 1 << 20, 1 << 24} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			keys, err := loadSOSDUint64(path, n)
			if err != nil {
				t.Skipf("SOSD data not available: %v (run bench/sosd_data/download.sh)", err)
			}
			t.Logf("loaded %d keys from osm_cellids, range [%d, %d]", len(keys), keys[0], keys[len(keys)-1])

			runTradeoffBench(t, benchConfig{
				distName:   "sosd_osm",
				n:          n,
				keys:       keys,
				queryCount: queryCount,
				queryFunc: func(rangeLen uint64, seed int64) [][2]uint64 {
					return generateSmartQueries(keys, queryCount, rangeLen, rand.New(rand.NewSource(seed)))
				},
				keySource: "sosd",
				keyFile:   "osm_cellids_800M_uint64",
				queryStrategy: "smart_mix",
				queryStrategyParams: map[string]interface{}{
					"nearKeyWeight": queryWeightNearKey,
					"inGapWeight":   queryWeightInGap,
					"uniformWeight": queryWeightUniform,
				},
			})
		})
	}
}
