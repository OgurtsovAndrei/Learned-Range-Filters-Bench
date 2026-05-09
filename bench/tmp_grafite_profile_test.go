package bench_test

import (
	"Thesis-bench-industry/thirdparty/grafite"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestGrafiteClusteredProfile(t *testing.T) {
	n := 1 << 24
	eps := 0.01

	// Load keys
	keys, err := loadSOSDUint64(syntheticDataPath(syntheticFile("clustered", n)), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) > n {
		keys = keys[:n]
	}

	// Build filter
	fmt.Println("Building Grafite filter...")
	start := time.Now()
	f := grafite.NewWithEpsL(keys, eps, 1)
	fmt.Printf("Build took %v. Size: %v bits\n", time.Since(start), f.SizeInBits())

	// Generate some queries
	rng := rand.New(rand.NewSource(12345))
	var queries [][2]uint64
	for i := 0; i < 10000; i++ {
		k := keys[rng.Intn(len(keys))] + 1
		queries = append(queries, [2]uint64{k, k})
	}

	fmt.Println("Running queries...")
	start = time.Now()
	res := f.QueryBatch(queries)
	dur := time.Since(start)

	fps := 0
	for _, r := range res {
		if !r { // expected all to be present, so this is just to use the result
			fps++
		}
	}
	
	fmt.Printf("Queries took %v (%.1f ns/op). FPs: %d\n", dur, float64(dur.Nanoseconds())/float64(len(queries)), fps)
}
