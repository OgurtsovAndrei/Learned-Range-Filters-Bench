package bucket

import (
	"Thesis/bits"
	"Thesis/utils/errutil"
	"Thesis-bench-industry/experiments/mmph/go-boomphf-bs/inline-uint64"
	"Thesis-bench-industry/experiments/trie/azft"
	"Thesis-bench-industry/experiments/trie/zft"
	"Thesis/utils"
	"fmt"
	"math"
	"math/rand"
	"time"
	"unsafe"
)

// Bucket represents a single bucket in the hash structure
type Bucket struct {
	mphf      *inline_uint64.H // MPHF for this bucket
	ranks     []uint8          // Local ranks within this bucket
	delimiter bits.BitString   // The delimiter (last key) of this bucket
}

// MonotoneHashWithTrie implements monotone minimal perfect hashing using
// approximate z-fast trie for relative ranking as described in Section 5 of the MMPH paper.
// This approach uses a relative trie on bucket delimiters for more efficient ranking.
// E, S, I are the type parameters for the underlying ApproxZFastTrie.
type MonotoneHashWithTrie[E zft.UNumber, S zft.UNumber, I zft.UNumber] struct {
	bucketSize int

	// Approximate Z-Fast Trie for relative ranking of bucket delimiters
	delimiterTrie *azft.ApproxZFastTrie[E, S, I]

	// Efficient bucket storage with signatures instead of full keys
	buckets []*Bucket

	// Statistics
	TrieRebuildAttempts int // Number of trie rebuild attempts during construction
}

const maxTrieRebuilds = 100 // Maximum number of attempts to build a working trie

// NewMonotoneHashWithTrie creates a new monotone hash function using approximate z-fast trie
// for bucket identification. The buckets are divided based on lexicographic order,
// with delimiters being the last key in each bucket.
// It validates that all keys work correctly with the trie and rebuilds with new seeds if needed.
// Uses a random seed from time.Now() for the trie construction.
//
// Type-parameter sizing notes:
//   - E (extent length) must represent max key length in bits: E >= ceil(log2(w)).
//   - I (delimiter-trie node index) must represent node indices plus sentinel.
//     For m delimiters, a binary trie has <= 2m-1 nodes, so 2m is a safe upper bound.
//   - S (PSig width) follows probabilistic trie analysis:
//     S >= log2(log2(w)) + log2(1/epsilon_query).
//     In the relative-trie setting (Theorem 5.2), epsilon_query = m/n, giving
//     S >= log2(log2(w)) + log2(n/m).
//     With bucket size b (so m ~= n/b), this is log2(log2(w)) + log2(b):
//     if b=log n then this becomes Theta(log log n + log log w).
//
// References:
//   - papers/MonotoneMinimalPerfectHashing.pdf
//   - papers/MMPH/Definitions-and-Tools.md
//   - papers/MMPH/Section-3-Bucketing.md
//   - papers/MMPH/Section-4-Relative-Ranking.md
//   - papers/MMPH/Section-5-Relative-Trie.md
func NewMonotoneHashWithTrie[E zft.UNumber, S zft.UNumber, I zft.UNumber](data []bits.BitString) (*MonotoneHashWithTrie[E, S, I], error) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return NewMonotoneHashWithTrieSeeded[E, S, I](data, rng.Uint64())
}

// NewMonotoneHashWithTrieSeeded creates a new monotone hash function using approximate z-fast trie
// with a specified seed for deterministic trie construction.
// The buckets are divided based on lexicographic order, with delimiters being the last key in each bucket.
// It validates that all keys work correctly with the trie and rebuilds with different derived seeds if needed
// (Las Vegas approach to ensure 100% correctness for the input set).
//
// Type-parameter sizing notes are the same as in NewMonotoneHashWithTrie.
func NewMonotoneHashWithTrieSeeded[E zft.UNumber, S zft.UNumber, I zft.UNumber](data []bits.BitString, baseSeed uint64) (*MonotoneHashWithTrie[E, S, I], error) {
	if len(data) == 0 {
		return &MonotoneHashWithTrie[E, S, I]{}, nil
	}

	// Choose bucket size as log n (as suggested in the paper)
	minBucketSize := max(int(math.Ceil(math.Log2(float64(len(data))))), 1)
	if minBucketSize > 256 {
		errutil.Bug("bucketSize must be <= 256 when using uint8 optimization")
	}
	bucketSize := 256 // max value of uint8

	n := len(data)
	numBuckets := (n + bucketSize - 1) / bucketSize

	mh := &MonotoneHashWithTrie[E, S, I]{
		bucketSize: bucketSize,
		buckets:    make([]*Bucket, numBuckets),
	}

	var delimiters []bits.BitString
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

			bucketMPHF := inline_uint64.NewDefault(bucketHashes)

			for j, h := range bucketHashes {
				if idx := bucketMPHF.Query(h); idx == 0 {
					errutil.Bug("boomphf failed immediately on construction for bucket %d key %d", i, j)
				}
			}

			bucketRanks := make([]uint8, len(bucketKeys))
			for localRank, k := range bucketKeys {
				phfIdx := bucketMPHF.Query(k.Hash()) - 1
				bucketRanks[phfIdx] = uint8(localRank)
			}

			delimiter := bucketKeys[len(bucketKeys)-1]
			delimiters = append(delimiters, delimiter)

			mh.buckets[i] = &Bucket{
				mphf:      bucketMPHF,
				ranks:     bucketRanks,
				delimiter: delimiter,
			}
		}
	}

	if len(delimiters) > 0 {
		err := mh.buildValidatedTrieWithIndices(data, delimiters, baseSeed)
		if err != nil {
			return nil, err
		}
	}

	return mh, nil
}

// buildValidatedTrieWithIndices builds the approximate z-fast trie with bucket indices and validates it works for all keys.
// It retries with different derived seeds if validation fails.
// This is a Las Vegas approach to handle the probabilistic nature of the AZFT (FPs and FNs).
// Since MMPH only needs to be correct for its build set, we ensure 100% runtime correctness
// by validating at build time. See mmph/relative_trie/README.md for details.
func (mh *MonotoneHashWithTrie[E, S, I]) buildValidatedTrieWithIndices(allKeys []bits.BitString, delimiters []bits.BitString, baseSeed uint64) error {

	for attempt := 0; attempt < maxTrieRebuilds; attempt++ {
		mh.TrieRebuildAttempts = attempt + 1

		trySeed := baseSeed + uint64(attempt)

		var err error
		mh.delimiterTrie, err = azft.NewApproxZFastTrieWithSeed[E, S, I](delimiters, trySeed)
		if err != nil {
			return err
		}

		if mh.validateAllKeys(allKeys) {
			return nil
		}
	}

	return fmt.Errorf("failed to build working approximate z-fast trie after %d attempts, try to increase S and/or I", maxTrieRebuilds)
}

// validateAllKeys checks that the trie can correctly handle all input keys
// Uses two-pointer approach since both keys and buckets are sorted
// Reports statistics on which candidates match at the end
func (mh *MonotoneHashWithTrie[E, S, I]) validateAllKeys(allKeys []bits.BitString) bool {
	if len(allKeys) == 0 || len(mh.buckets) == 0 {
		return true
	}

	bucketIdx := 0
	maxDelimiterIndex := I(^I(0))

	// Statistics tracking
	cand1Matches := 0
	cand2Matches := 0
	cand3Matches := 0
	cand4Matches := 0
	cand5Matches := 0
	cand6Matches := 0

	for _, key := range allKeys {
		for bucketIdx < len(mh.buckets) {
			bucket := mh.buckets[bucketIdx]
			if bucket != nil && key.Compare(bucket.delimiter) <= 0 {
				break
			}
			bucketIdx++
		}

		if bucketIdx >= len(mh.buckets) {
			errutil.Bug("bucketIdx out of range")
		}

		expectedBucket := bucketIdx

		cand1, cand2, cand3, cand4, cand5, cand6 := mh.delimiterTrie.LowerBound(key)

		foundCorrectBucket := false

		candidates := []*azft.NodeData[E, S, I]{cand1, cand2, cand3, cand4, cand5, cand6}
		matchCounts := []*int{&cand1Matches, &cand2Matches, &cand3Matches, &cand4Matches, &cand5Matches, &cand6Matches}

		for i, candidate := range candidates {
			if candidate == nil {
				continue
			}

			if candidate.Rank != maxDelimiterIndex {
				candidateBucketIdx := int(candidate.Rank)
				if candidateBucketIdx == expectedBucket {
					foundCorrectBucket = true
					*matchCounts[i]++
				}
			}
		}

		if !foundCorrectBucket {
			return false
		}
	}

	return true
}

// GetRank returns the rank of the given key in the original sorted order.
// It uses the approximate z-fast trie to find the correct bucket, then
// computes the local rank within that bucket.
func (mh *MonotoneHashWithTrie[E, S, I]) GetRank(key bits.BitString) int {
	if mh.delimiterTrie == nil || len(mh.buckets) == 0 {
		return -1
	}

	cand1, cand2, cand3, cand4, cand5, cand6 := mh.delimiterTrie.LowerBound(key)

	bucketIdx := -1
	maxDelimiterIndex := I(^I(0))

	tryCandidate := func(candidate *azft.NodeData[E, S, I]) bool {
		if candidate == nil {
			return false
		}

		if candidate.Rank != maxDelimiterIndex {
			candidateBucketIdx := int(candidate.Rank)

			if candidateBucketIdx < len(mh.buckets) && mh.buckets[candidateBucketIdx] != nil {
				bucket := mh.buckets[candidateBucketIdx]

				if key.Compare(bucket.delimiter) <= 0 {
					if candidateBucketIdx == 0 ||
						(candidateBucketIdx > 0 && mh.buckets[candidateBucketIdx-1] != nil &&
							key.Compare(mh.buckets[candidateBucketIdx-1].delimiter) > 0) {
						bucketIdx = candidateBucketIdx
						return true
					}
				}
			}
		}
		return false
	}

	if !tryCandidate(cand1) {
		if !tryCandidate(cand2) {
			if !tryCandidate(cand3) {
				if !tryCandidate(cand4) {
					if !tryCandidate(cand5) {
						tryCandidate(cand6)
					}
				}
			}
		}
	}

	if bucketIdx == -1 || bucketIdx >= len(mh.buckets) || mh.buckets[bucketIdx] == nil {
		return -1
	}

	bucket := mh.buckets[bucketIdx]
	keyHash := key.Hash()
	localPhfIdx := bucket.mphf.Query(keyHash)
	if localPhfIdx == 0 || int(localPhfIdx) > len(bucket.ranks) {
		return -1
	}

	localOffset := int(bucket.ranks[localPhfIdx-1])

	return bucketIdx*mh.bucketSize + localOffset
}

// Size returns the total size of the structure in bytes.
//
// Approximate memory model (bits):
//   - m = ceil(n/256), where n is the number of keys.
//   - MHT_bits ~= O(1) + AZFT_bits + MPHF_bucket_bits + 8*n + m*(W+8),
//     where W is key length in bits, MPHF_bucket_bits is often approximated as ~3*n.
//   - With MPHF_bucket_bits ~= 3*n:
//     MHT_bits ~= O(1) + AZFT_bits + (3+8)*n + m*(W+8).
//
// Notes:
//   - This is an asymptotic/engineering model, not byte-exact (allocator overhead,
//     slice headers, and alignment are ignored in the formula).
//   - AZFT_bits is detailed in ApproxZFastTrie.ByteSize comments.
func (mh *MonotoneHashWithTrie[E, S, I]) Size() int {
	size := 4

	if mh.delimiterTrie != nil {
		size += mh.delimiterTrie.ByteSize()
	}

	for _, bucket := range mh.buckets {
		if bucket != nil {
			size += bucket.mphf.Size()
			size += len(bucket.ranks)
			size += int(bucket.delimiter.Size())/8 + 1
		}
	}

	size += 4

	return size
}

// ByteSize returns the total size of the structure in bytes (same as Size for consistency).
func (mh *MonotoneHashWithTrie[E, S, I]) ByteSize() int {
	return mh.Size()
}

// MemDetailed returns a detailed memory usage report for MonotoneHashWithTrie.
func (mh *MonotoneHashWithTrie[E, S, I]) MemDetailed() utils.MemReport {
	if mh == nil {
		return utils.MemReport{Name: "MonotoneHashWithTrie", TotalBytes: 0}
	}

	headerSize := int(unsafe.Sizeof(*mh))
	trieReport := mh.delimiterTrie.MemDetailed()

	bucketsReport := utils.MemReport{Name: "buckets", TotalBytes: 0}
	for i, bucket := range mh.buckets {
		if bucket == nil {
			continue
		}
		bucketSize := int(unsafe.Sizeof(*bucket))
		mphSize := bucket.mphf.Size()
		ranksSize := len(bucket.ranks)
		delimSize := int(bucket.delimiter.Size())/8 + 1

		totalBucketSize := bucketSize + mphSize + ranksSize + delimSize
		bucketsReport.TotalBytes += totalBucketSize

		bucketsReport.Children = append(bucketsReport.Children, utils.MemReport{
			Name:       fmt.Sprintf("bucket_%d", i),
			TotalBytes: totalBucketSize,
			Children: []utils.MemReport{
				{Name: "header", TotalBytes: bucketSize},
				{Name: "mphf", TotalBytes: mphSize},
				{Name: "ranks", TotalBytes: ranksSize},
				{Name: "delimiter", TotalBytes: delimSize},
			},
		})
	}

	return utils.MemReport{
		Name:       "MonotoneHashWithTrie",
		TotalBytes: mh.ByteSize(),
		Children: []utils.MemReport{
			{Name: "header", TotalBytes: headerSize},
			trieReport,
			bucketsReport,
		},
	}
}
