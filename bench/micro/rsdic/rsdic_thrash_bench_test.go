package rsdic_test

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// BenchmarkRsdicSelect1WithThrash measures rsdic.Select1 latency on the
// real SODA-FB inner rsdic, but interleaves each call with N random
// byte reads from a separately-allocated "thrash" array. The thrash
// array models the 48 MB ExactRangeEmptiness.packedData that the SODA
// query path touches between Select calls via the bucket binary search.
//
// If rsdic.Select1 latency rises from ~67 ns (isolated) toward the ~1585
// ns figure pprof attributes to it under the full SODA pipeline as the
// thrash size grows, then the observation file's hypothesis — that the
// rsdic latency inflation is interaction-driven cache thrashing rather
// than intrinsic — is confirmed.
//
// One b.N iteration = one Select1 + thrashHopsPerSelect random reads
// from the thrash array.
func BenchmarkRsdicSelect1WithThrash(b *testing.B) {
	const n = 1 << 24
	thrashSizes := []int{
		0,                // baseline: pure isolated Select1
		1 << 20,          // 1  MB — fits in L2
		4 << 20,          // 4  MB — fits in L2
		16 << 20,         // 16 MB — at L2 edge
		48 << 20,         // 48 MB — packedData scale
		192 << 20,        // 192 MB — far past any cache
	}
	const thrashHopsPerSelect = 14 // matches binsearch depth at L=65536

	path := rsdicCachePath("sosd_fb", n, 65536)
	if _, err := os.Stat(path); err != nil {
		b.Skipf("missing dump %s — run TestDumpSodaRSDic first", path)
	}
	rsd := loadRSDic(b, path)
	ones := rsd.OneNum()

	// Pre-generate the rank stream (200K random ranks).
	const kIters = 200_000
	rng := rand.New(rand.NewSource(7))
	ranks := make([]uint64, kIters)
	for i := range ranks {
		ranks[i] = uint64(rng.Int63n(int64(ones)))
	}

	for _, ts := range thrashSizes {
		ts := ts
		b.Run(fmt.Sprintf("thrash=%dMB", ts>>20), func(b *testing.B) {
			var thrash []byte
			if ts > 0 {
				thrash = make([]byte, ts)
				// Touch every page once so the OS commits physical memory.
				for i := 0; i < ts; i += 4096 {
					thrash[i] = byte(i)
				}
			}
			thrashMask := 0
			if ts > 0 {
				thrashMask = ts - 1 // power-of-2 mask for cheap modulo
			}

			// Pre-generate the thrash offsets — separate stream so the
			// indirection cost is constant per iteration.
			offsets := make([]int, kIters*thrashHopsPerSelect)
			rng2 := rand.New(rand.NewSource(99))
			for i := range offsets {
				offsets[i] = rng2.Int() & thrashMask
			}

			b.ResetTimer()
			var sink uint64
			var bsink byte
			for i := 0; i < b.N; i++ {
				sink ^= rsd.Select1(ranks[i%kIters])
				if ts > 0 {
					base := (i * thrashHopsPerSelect) % len(offsets)
					for h := 0; h < thrashHopsPerSelect; h++ {
						bsink ^= thrash[offsets[base+h]]
					}
				}
			}
			b.StopTimer()
			if sink == 0xDEADBEEF || bsink == 0xFF {
				b.Log("sink trick")
			}
			// Per-Select1 cost: divide ns/op by 1 (one Select1 per iter)
			// — but the thrash hops are also charged against b.N. Report
			// the thrash overhead separately so the user can subtract.
			b.ReportMetric(float64(thrashHopsPerSelect), "thrash_hops_per_op")
		})
	}
}
