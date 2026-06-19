package bucket

import (
	"Thesis/utils/errutil"
	"Thesis-bench-industry/experiments/mmph/go-boomphf-bs/inline-uint64"
	"fmt"
	"math"

	"Thesis/bits"
	"Thesis/bits/maps"
)

type MonotoneHash struct {
	bucketSize int

	// d0: KeyHash -> LCP Length (in bits)
	d0Table   *inline_uint64.H // overhead of 2-3 bits per key
	d0Lengths []uint16         // O(n * log w) bits

	// d1: PrefixHash -> Bucket Index
	d1Table   *inline_uint64.H // overhead of 2-3 bits per bucket
	d1Indices []int32          // O((n/b) * log(n/b)) <= O(n) bits, when b = log n

	// buckets: KeyHash -> Local Rank
	buckets     []*inline_uint64.H // overhead of 2-3 bits per key
	bucketRanks [][]uint8          // O(n log(b)) = O(b log(log(n))) bits, when b = log n
}

func NewMonotoneHash(data []bits.BitString) *MonotoneHash {
	if len(data) == 0 {
		return &MonotoneHash{}
	}

	bucketSize := max(int(math.Ceil(math.Log2(float64(len(data))))), 1)

	if bucketSize > 256 {
		panic("bucketSize must be <= 256 when using uint8 optimization")
	}

	n := len(data)
	numBuckets := (n + bucketSize - 1) / bucketSize

	mh := &MonotoneHash{
		bucketSize:  bucketSize,
		buckets:     make([]*inline_uint64.H, numBuckets),
		bucketRanks: make([][]uint8, numBuckets),
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

			mh.buckets[i] = inline_uint64.NewDefault(bucketHashes)

			for j, h := range bucketHashes {
				if idx := mh.buckets[i].Query(h); idx == 0 {
					panic(fmt.Sprintf("boomphf failed immediately on construction for bucket %d key %d", i, j))
				}
			}

			mh.bucketRanks[i] = make([]uint8, len(bucketKeys))
			for localRank, k := range bucketKeys {
				phfIdx := mh.buckets[i].Query(k.Hash()) - 1
				mh.bucketRanks[i][phfIdx] = uint8(localRank)
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

			// Wait, lcpSize should be the size of the LCP bits
			actualLcpSize := int(lcp.Size())
			for _, k := range bucketKeys {
				keyToLcpLen.Put(k, actualLcpSize)
			}
			allKeys = append(allKeys, bucketKeys...)
		}
	}

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
			lcpSize, _ := keyToLcpLen.Get(k)
			mh.d0Lengths[phfIdx] = uint16(lcpSize)
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

	return mh
}

func (mh *MonotoneHash) GetRank(key bits.BitString) int {
	if mh.d0Table == nil {
		return -1
	}

	keyHash := key.Hash()

	d0PhfIdx := mh.d0Table.Query(keyHash)
	if d0PhfIdx == 0 || int(d0PhfIdx) > len(mh.d0Lengths) {
		return -1
	}
	lcpLen := int(mh.d0Lengths[d0PhfIdx-1])

	if int(key.Size()) < lcpLen {
		return -1
	}

	prefix := key.Prefix(lcpLen)
	prefixHash := prefix.Hash()

	d1PhfIdx := mh.d1Table.Query(prefixHash)
	if d1PhfIdx == 0 || int(d1PhfIdx) > len(mh.d1Indices) {
		errutil.Bug("d1Table miss for prefix %s (key %s). Likely hash collision or unsorted input.", prefix.PrettyString(), key.PrettyString())
		return -1
	}
	bucketIdx := int(mh.d1Indices[d1PhfIdx-1])

	if bucketIdx >= len(mh.buckets) || mh.buckets[bucketIdx] == nil {
		return -1
	}

	localPhfIdx := mh.buckets[bucketIdx].Query(keyHash)
	if localPhfIdx == 0 || int(localPhfIdx) > len(mh.bucketRanks[bucketIdx]) {
		errutil.Bug("Local bucket %d miss for key %s", bucketIdx, key.PrettyString())
		return -1
	}

	localOffset := int(mh.bucketRanks[bucketIdx][localPhfIdx-1])

	return bucketIdx*mh.bucketSize + localOffset
}

// Size returns the total size of the structure in bytes.
// It accounts for:
// - d0Table (MPHF) and d0Lengths (array)
// - d1Table (MPHF) and d1Indices (array)
// - buckets (MPHF per bucket) and bucketRanks (array per bucket)
func (mh *MonotoneHash) Size() int {
	var size = 4 // bucket size

	if mh.d0Table != nil {
		size += mh.d0Table.Size()
	}
	size += len(mh.d0Lengths) * 2

	if mh.d1Table != nil {
		size += mh.d1Table.Size()
	}
	size += len(mh.d1Indices) * 4

	for _, b := range mh.buckets {
		if b != nil {
			size += b.Size()
		}
	}
	for _, r := range mh.bucketRanks {
		size += len(r) * 1
	}
	return size
}

// ByteSize returns the total size of the structure in bytes (same as Size for consistency).
func (mh *MonotoneHash) ByteSize() int {
	return mh.Size()
}
