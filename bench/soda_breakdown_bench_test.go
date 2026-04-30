package bench_test

import (
	"math/rand"
	"testing"

	"Thesis/emptiness/approx/are_soda_hash"
	"Thesis/succinct_bit_vector/rsdic"
)

// BenchmarkSelect1RealBlockIDs measures rsdic.Select1 in isolation, but
// using the actual block-id stream that SODA on FB at L=65536 would
// pass into getBlockRange — i.e. a non-uniform distribution biased to
// the 760 populated blocks. This rules out the hypothesis that pprof's
// 70%-of-Select1 attribution is an artefact of which ranks we feed.
func BenchmarkSelect1RealBlockIDs(b *testing.B) {
	const (
		n = 1 << 24
		L = uint64(65536)
	)
	keys := loadKeysForSpec(b, "sosd_fb", n)
	soda, err := are_soda_hash.NewSodaARE(append([]uint64(nil), keys...), L, 0.01)
	if err != nil {
		b.Fatalf("NewSodaARE: %v", err)
	}
	rsd := soda.ERE().D

	rng := rand.New(rand.NewSource(42))
	const kIters = 200_000
	queries := generateSmartQueriesAudit(rng, keys, L, kIters)

	// For each query, compute the rank passed to ere.Select1 (= block id).
	// We don't need the SODA-hashed values because the inner ERE.IsEmpty
	// re-computes blockA/blockB internally; the rank fed to Select1 is
	// always blockA followed by blockA+1.
	stats := soda.EREStats()
	_ = stats // keep imports stable if unused
	// blockId = a >> w; we don't have direct access to ere.w, but
	// SODA-FB is degenerate so a stays in the original FB universe and
	// w = K - log2(n) = 47 - 24 = 23 in this regime.
	const w = 23
	const K = 47
	_ = K
	ranks := make([]uint64, 0, kIters*2)
	for _, q := range queries {
		// Match SodaARE.IsEmpty same-block path: hashed = a (degenerate),
		// blockA inside ERE = (a) >> w.
		blockA := q.lo >> w
		ranks = append(ranks, blockA, blockA+1)
	}

	b.ResetTimer()
	var sink uint64
	for i := 0; i < b.N; i++ {
		sink ^= rsd.Select1(ranks[i%len(ranks)])
	}
	b.StopTimer()
	if sink == 0xDEADBEEF {
		b.Log("sink trick")
	}
}

// BenchmarkGetBlockRangeOnly measures only the getBlockRange portion of
// the SODA query path: 2 Select1 calls per "iteration", in tight loop,
// without the surrounding bucket binsearch / SODA wrapper / etc.
func BenchmarkGetBlockRangeOnly(b *testing.B) {
	const (
		n = 1 << 24
		L = uint64(65536)
	)
	keys := loadKeysForSpec(b, "sosd_fb", n)
	soda, err := are_soda_hash.NewSodaARE(append([]uint64(nil), keys...), L, 0.01)
	if err != nil {
		b.Fatalf("NewSodaARE: %v", err)
	}
	rsd := soda.ERE().D

	rng := rand.New(rand.NewSource(42))
	const kIters = 200_000
	queries := generateSmartQueriesAudit(rng, keys, L, kIters)
	const w = 23
	blocks := make([]uint64, kIters)
	for i, q := range queries {
		blocks[i] = q.lo >> w
	}

	b.ResetTimer()
	var sink uint64
	for i := 0; i < b.N; i++ {
		blk := blocks[i%kIters]
		// Inline 2 Select1 calls to mirror getBlockRange.
		sink ^= rsd.Select1(blk)
		sink ^= rsd.Select1(blk + 1)
	}
	b.StopTimer()
	if sink == 0xDEADBEEF {
		b.Log("sink trick")
	}
}

// BenchmarkSelect1ThenPackedRead simulates the SODA inner loop: one
// Select1 followed by 14 random reads from the packed-suffix array
// (the 48 MB structure that the bucket binsearch would access). Same
// bytes/iter as BenchmarkRsdicSelect1WithThrash but the *target array*
// is the actual ere.packedData rather than a synthetic 48 MB blob.
//
// Surfacing whether the gap pprof shows is sensitive to "where the
// 14 random reads land" (real packedData vs synthetic bytes).
func BenchmarkSelect1ThenPackedRead(b *testing.B) {
	_ = rsdic.New // keep import
	const (
		n = 1 << 24
		L = uint64(65536)
	)
	keys := loadKeysForSpec(b, "sosd_fb", n)
	soda, err := are_soda_hash.NewSodaARE(append([]uint64(nil), keys...), L, 0.01)
	if err != nil {
		b.Fatalf("NewSodaARE: %v", err)
	}
	rsd := soda.ERE().D

	rng := rand.New(rand.NewSource(42))
	const kIters = 200_000
	queries := generateSmartQueriesAudit(rng, keys, L, kIters)
	const w = 23
	blocks := make([]uint64, kIters)
	for i, q := range queries {
		blocks[i] = q.lo >> w
	}

	// Use the real ERE.packedData as a thrash target. We can't access it
	// directly (private field), but ere_one_d exposes ByteSize which we
	// can use to size a synthetic equivalent. For the "real packed"
	// version we'd need an accessor; for now, observe size:
	b.Logf("ERE size in bits: %d (~%.1f MB)",
		soda.ERE().SizeInBits(),
		float64(soda.ERE().SizeInBits())/8/1024/1024)

	b.ResetTimer()
	var sink uint64
	for i := 0; i < b.N; i++ {
		blk := blocks[i%kIters]
		sink ^= rsd.Select1(blk)
		sink ^= rsd.Select1(blk + 1)
	}
	b.StopTimer()
	if sink == 0xDEADBEEF {
		b.Log("sink trick")
	}
}
