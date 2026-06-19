package bucket

import (
	"Thesis/utils/errutil"
	"Thesis-bench-industry/experiments/mmph/go-boomphf-bs/inline-uint64"
	"fmt"
	"math"

	"Thesis/bits"
	"Thesis/bits/maps"
)

type DebugMonotoneHash struct {
	bucketSize int

	// d0: KeyHash -> LCP Length (in bits)
	d0Table    *inline_uint64.H
	d0Lengths  []uint16
	d0DebugMap *maps.BitMap[uint16] // DEBUG

	// d1: PrefixHash -> Bucket Index
	d1Table    *inline_uint64.H
	d1Indices  []int32
	d1DebugMap *maps.BitMap[int32] // DEBUG

	// buckets: KeyHash -> Local Rank
	buckets         []*inline_uint64.H
	bucketRanks     [][]uint8
	bucketsDebugMap []*maps.BitMap[uint8] // DEBUG
}

func NewDebugMonotoneHash(data []bits.BitString) *DebugMonotoneHash {
	if len(data) == 0 {
		return &DebugMonotoneHash{}
	}

	bucketSize := max(int(math.Ceil(math.Log2(float64(len(data))))), 1)

	if bucketSize > 256 {
		panic("bucketSize must be <= 256 when using uint8 optimization")
	}

	n := len(data)
	numBuckets := (n + bucketSize - 1) / bucketSize

	mh := &DebugMonotoneHash{
		bucketSize:      bucketSize,
		buckets:         make([]*inline_uint64.H, numBuckets),
		bucketRanks:     make([][]uint8, numBuckets),
		bucketsDebugMap: make([]*maps.BitMap[uint8], numBuckets), // DEBUG
		d1DebugMap:      maps.NewBitMap[int32](),                 // DEBUG
		d0DebugMap:      maps.NewBitMap[uint16](),                // DEBUG
	}

	var allKeys []bits.BitString
	var allLcps []bits.BitString

	keyToLcpLen := maps.NewBitMap[int]()
	prefixToBucketIdx := maps.NewBitMap[int]()

	for i := 0; i < numBuckets; i++ {
		start := i * bucketSize
		end := start + bucketSize
		if end > n {
			end = n
		}

		bucketKeys := data[start:end]

		if len(bucketKeys) > 0 {
			bucketHashes := make([]uint64, len(bucketKeys))
			for j, k := range bucketKeys {
				bucketHashes[j] = k.Hash()
			}

			gamma := 2.0
			if len(bucketKeys) < 10 {
				gamma = 10.0
			}
			mh.buckets[i] = inline_uint64.New(gamma, bucketHashes)

			for j, h := range bucketHashes {
				if idx := mh.buckets[i].Query(h); idx == 0 {
					panic(fmt.Sprintf("boomphf failed immediately on construction for bucket %d key %d", i, j))
				}
			}

			mh.bucketsDebugMap[i] = maps.NewBitMap[uint8]()

			mh.bucketRanks[i] = make([]uint8, len(bucketKeys))
			for localRank, k := range bucketKeys {
				phfIdx := mh.buckets[i].Query(k.Hash()) - 1
				mh.bucketRanks[i][phfIdx] = uint8(localRank)
				mh.bucketsDebugMap[i].Put(k, uint8(localRank))
			}

			lcp := bucketKeys[0]
			for _, k := range bucketKeys[1:] {
				lcpLen := lcp.GetLCPLength(k)
				lcp = lcp.Prefix(int(lcpLen))
			}

			if _, exists := prefixToBucketIdx.Get(lcp); !exists {
				allLcps = append(allLcps, lcp)
				prefixToBucketIdx.Put(lcp, i)
			}

			lcpSize := int(lcp.Size())
			for _, k := range bucketKeys {
				keyToLcpLen.Put(k, lcpSize)
			}
			allKeys = append(allKeys, bucketKeys...)
		}
	}

	prefixToBucketIdx.Range(func(p bits.BitString, idx int) bool {
		mh.d1DebugMap.Put(p, int32(idx))
		return true
	})

	if len(allKeys) > 0 {
		allKeyHashes := make([]uint64, len(allKeys))
		for i, k := range allKeys {
			allKeyHashes[i] = k.Hash()
		}
		mh.d0Table = inline_uint64.NewDefault(allKeyHashes)

		mh.d0Lengths = make([]uint16, len(allKeys))
		for _, k := range allKeys {
			phfIdx := mh.d0Table.Query(k.Hash()) - 1
			if phfIdx+1 == 0 {
				panic("boomphf construction failure: Query returned 0 for known key in d0Table")
			}
			lcpLen, _ := keyToLcpLen.Get(k)
			mh.d0Lengths[phfIdx] = uint16(lcpLen)
			mh.d0DebugMap.Put(k, uint16(lcpLen))
		}
	}

	if len(allLcps) > 0 {
		lcpHashes := make([]uint64, len(allLcps))
		for i, p := range allLcps {
			lcpHashes[i] = p.Hash()
		}
		mh.d1Table = inline_uint64.NewDefault(lcpHashes)

		mh.d1Indices = make([]int32, len(allLcps))
		for _, p := range allLcps {
			phfIdx := mh.d1Table.Query(p.Hash()) - 1
			idx, _ := prefixToBucketIdx.Get(p)
			mh.d1Indices[phfIdx] = int32(idx)
		}
	}

	fmt.Println("=== DEBUG: Constructed DebugMonotoneHash Maps ===")
	fmt.Println("--- D0 (Key -> LCP Length) ---")
	mh.d0DebugMap.Range(func(k bits.BitString, v uint16) bool {
		fmt.Printf("Key: %s, Hash: %x, LCP Len: %d\n", k.PrettyString(), k.Hash(), v)
		return true
	})
	fmt.Println("--- D1 (Prefix -> Bucket Index) ---")
	mh.d1DebugMap.Range(func(k bits.BitString, v int32) bool {
		fmt.Printf("Prefix: %s, Hash: %x, Bucket: %d\n", k.PrettyString(), k.Hash(), v)
		return true
	})
	fmt.Println("--- Buckets (Key -> Local Rank) ---")
	for i, m := range mh.bucketsDebugMap {
		fmt.Printf("Bucket %d:\n", i)
		m.Range(func(k bits.BitString, v uint8) bool {
			fmt.Printf("  Key: %s, Hash: %x, Rank: %d\n", k.PrettyString(), k.Hash(), v)
			return true
		})
	}
	fmt.Println("============================================")

	return mh
}

func (mh *DebugMonotoneHash) GetRank(key bits.BitString) int {
	fmt.Printf("\n--- GetRank for %s (Hash: %x) ---\n", key.PrettyString(), key.Hash())

	if mh.d0Table == nil {
		return -1
	}

	keyHash := key.Hash()

	d0PhfIdx := mh.d0Table.Query(keyHash)
	lcpLen := -1
	if d0PhfIdx != 0 && int(d0PhfIdx) <= len(mh.d0Lengths) {
		lcpLen = int(mh.d0Lengths[d0PhfIdx-1])
	}
	expectedLcpLen, _ := mh.d0DebugMap.Get(key)
	fmt.Printf("1. D0 Lookup: Hash=%x -> PHF_Idx=%d -> LCP_Len=%d (Expected: %d)\n", keyHash, d0PhfIdx, lcpLen, expectedLcpLen)

	if d0PhfIdx == 0 || int(d0PhfIdx) > len(mh.d0Lengths) {
		errutil.BugOn(true, "d0Table miss for key %s", key.PrettyString())
		return -1
	}

	expectedLcpLen, ok := mh.d0DebugMap.Get(key)
	if !ok {
		errutil.BugOn(true, "DEBUG: Key %s not found in d0DebugMap!", key.PrettyString())
	} else if uint16(lcpLen) != expectedLcpLen {
		errutil.BugOn(true, "DEBUG: d0 mismatch for key %s. PHF got %d, Map got %d", key.PrettyString(), lcpLen, expectedLcpLen)
	}

	if int(key.Size()) < lcpLen {
		return -1
	}

	prefix := key.Prefix(lcpLen)
	prefixHash := prefix.Hash()

	d1PhfIdx := mh.d1Table.Query(prefixHash)
	bucketIdx := -1
	if d1PhfIdx != 0 && int(d1PhfIdx) <= len(mh.d1Indices) {
		bucketIdx = int(mh.d1Indices[d1PhfIdx-1])
	}
	expectedBucketIdx, _ := mh.d1DebugMap.Get(prefix)
	fmt.Printf("2. D1 Lookup: Prefix=%s, Hash=%x -> PHF_Idx=%d -> Bucket=%d (Expected: %d)\n", prefix.PrettyString(), prefixHash, d1PhfIdx, bucketIdx, expectedBucketIdx)

	if d1PhfIdx == 0 || int(d1PhfIdx) > len(mh.d1Indices) {
		errutil.BugOn(true, "d1Table miss for prefix %s (key %s)", prefix.PrettyString(), key.PrettyString())
		return -1
	}

	expectedBucketIdx, ok = mh.d1DebugMap.Get(prefix)
	if !ok {
		errutil.BugOn(true, "DEBUG: Prefix %s not found in d1DebugMap! (key %s)", prefix.PrettyString(), key.PrettyString())
	} else if int32(bucketIdx) != expectedBucketIdx {
		errutil.BugOn(true, "DEBUG: d1 mismatch for prefix %s. PHF got %d, Map got %d", prefix.PrettyString(), bucketIdx, expectedBucketIdx)
	}

	if bucketIdx >= len(mh.buckets) || mh.buckets[bucketIdx] == nil {
		return -1
	}

	localPhfIdx := mh.buckets[bucketIdx].Query(keyHash)
	localOffset := -1
	if localPhfIdx != 0 && int(localPhfIdx) <= len(mh.bucketRanks[bucketIdx]) {
		localOffset = int(mh.bucketRanks[bucketIdx][localPhfIdx-1])
	}
	expectedOffset, _ := mh.bucketsDebugMap[bucketIdx].Get(key)
	fmt.Printf("3. Bucket %d Lookup: Hash=%x -> PHF_Idx=%d -> LocalRank=%d (Expected: %d)\n", bucketIdx, keyHash, localPhfIdx, localOffset, expectedOffset)

	if localPhfIdx == 0 || int(localPhfIdx) > len(mh.bucketRanks[bucketIdx]) {
		errutil.BugOn(true, "Local bucket %d miss for key %s", bucketIdx, key.PrettyString())
		return -1
	}

	expectedOffset, ok = mh.bucketsDebugMap[bucketIdx].Get(key)
	if !ok {
		errutil.BugOn(true, "DEBUG: Key %s not found in bucketsDebugMap[%d]!", key.PrettyString(), bucketIdx)
	} else if uint8(localOffset) != expectedOffset {
		errutil.BugOn(true, "DEBUG: Local rank mismatch for key %s in bucket %d. PHF got %d, Map got %d", key.PrettyString(), bucketIdx, localOffset, expectedOffset)
	}

	return bucketIdx*mh.bucketSize + localOffset
}
