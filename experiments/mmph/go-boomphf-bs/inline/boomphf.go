// Package inline is a fast perfect hash function for massive key sets
// adapted from boomphf-inline.go
package inline

import (
	"Thesis/utils/errutil"
	"math/bits"

	tbits "Thesis/bits"
)

// H stores level sizes inline with the bitvector data
type H struct {
	b     bitvector
	ranks []uint32
}

// New constructs a perfect hash function with inlined level sizes.
func New(gamma float64, keys []tbits.BitString) *H {
	errutil.BugOn(uint64(len(keys)) > uint64(^uint32(0)), "too many keys")
	var h H

	var level uint32
	size := uint32(gamma * float64(len(keys)))
	size = (size + 511) &^ 511 // 8-word alignment for data (512 bits)

	A := newbv(size)
	collide := newbv(size)

	var redo []tbits.BitString
	var levels []bitvector
	var levelSizes []uint32

	for len(keys) > 0 {
		for _, v := range keys {
			hash := v.Hash()
			h1, h2 := uint32(hash), uint32(hash>>32)
			idx := (h1 ^ bits.RotateLeft32(h2, int(level))) % size

			if collide.get(idx) == 1 {
				continue
			}
			if A.get(idx) == 1 {
				collide.set(idx)
				continue
			}
			A.set(idx)
		}

		bv := newbv(size)
		for _, v := range keys {
			hash := v.Hash()
			h1, h2 := uint32(hash), uint32(hash>>32)
			idx := (h1 ^ bits.RotateLeft32(h2, int(level))) % size

			if collide.get(idx) == 1 {
				redo = append(redo, v)
				continue
			}
			bv.set(idx)
		}
		levels = append(levels, bv)
		levelSizes = append(levelSizes, size)

		keys = redo
		redo = redo[:0]
		size = uint32(gamma * float64(len(keys)))
		size = (size + 511) &^ 511
		A.reset()
		collide.reset()
		level++
	}

	// Flatten with inlined sizes and 8-word alignment for data
	var totalWords uint32
	for _, lv := range levels {
		totalWords += 8 + uint32(len(lv))
	}
	h.b = make(bitvector, totalWords)
	var current uint32
	for i, lv := range levels {
		h.b[current] = uint64(levelSizes[i]) // Inline size
		// words 1-7 are padding to keep data aligned to 8-word boundary
		copy(h.b[current+8:], lv)
		current += 8 + uint32(len(lv))
	}

	h.computeRanks()

	return &h
}

func (h *H) computeRanks() {
	var pop uint32
	h.ranks = make([]uint32, 0, (len(h.b)+7)/8)

	offset := uint32(0)
	for offset < uint32(len(h.b)) {
		size := uint32(h.b[offset])
		dataWords := (size + 63) / 64

		// Process 8 words of header (only data counts, but we must maintain rank index)
		for i := 0; i < 8; i++ {
			if (offset+uint32(i))%8 == 0 {
				h.ranks = append(h.ranks, pop)
			}
			// Header words (0-7) contribute 0 to population
		}

		// Process data words
		for i := uint32(0); i < dataWords; i++ {
			idx := offset + 8 + i
			if idx%8 == 0 {
				h.ranks = append(h.ranks, pop)
			}
			pop += uint32(bits.OnesCount64(h.b[idx]))
		}

		offset += 8 + dataWords
	}
}

// Query returns the index of the key
func (h *H) Query(k tbits.BitString) uint64 {
	hash := k.Hash()
	h1, h2 := uint32(hash), uint32(hash>>32)

	var current uint32
	level := 0
	for current < uint32(len(h.b)) {
		size := uint32(h.b[current])
		idx := (h1 ^ bits.RotateLeft32(h2, level)) % size

		dataStart := current + 8
		n := dataStart + (idx / 64)

		if h.b[n]&(1<<(idx%64)) == 0 {
			current = dataStart + (size+63)/64
			level++
			continue
		}

		rank := h.ranks[n/8]
		// Since dataStart is 8-word aligned, n &^ 7 will never be less than dataStart
		// So we don't need to skip the size word here.
		for j := n &^ 7; j < n; j++ {
			rank += uint32(bits.OnesCount64(h.b[j]))
		}
		rank += uint32(bits.OnesCount64(h.b[n] << (64 - (idx % 64))))

		return uint64(rank + 1)
	}

	return 0
}

// Size returns the size in bytes
func (h *H) Size() int {
	return len(h.b)*8 + len(h.ranks)*4
}

// ByteSize returns the size in bytes (same as Size for consistency)
func (h *H) ByteSize() int {
	return h.Size()
}

type bitvector []uint64

func newbv(size uint32) bitvector {
	return make([]uint64, uint(size+63)/64)
}

// get bit 'bit' in the bitvector d
func (b bitvector) get(bit uint32) uint {
	shift := bit % 64
	bb := b[bit/64]
	bb &= (1 << shift)

	return uint(bb >> shift)
}

// set bit 'bit' in the bitvector d
func (b bitvector) set(bit uint32) {
	b[bit/64] |= (1 << (bit % 64))
}

func (b bitvector) reset() {
	for i := range b {
		b[i] = 0
	}
}
