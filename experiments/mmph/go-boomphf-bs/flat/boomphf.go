// Package flat is a fast perfect hash function for massive key sets
// adapted from boomphf-flat-arrays.go
package flat

import (
	"Thesis/utils/errutil"
	"math/bits"

	tbits "Thesis/bits"
)

// H is hash function data
type H struct {
	b          bitvector
	ranks      []uint32
	levelSizes []uint32 // Size of bitvector for each level (in bits)
}

// New constructs a perfect hash function for the keys. The gamma value controls the space used.
func New(gamma float64, keys []tbits.BitString) *H {
	errutil.BugOn(uint64(len(keys)) > uint64(^uint32(0)), "too many keys")
	var h H

	var level uint32

	size := uint32(gamma * float64(len(keys)))
	size = (size + 63) &^ 63

	A := newbv(size)
	collide := newbv(size)

	var redo []tbits.BitString
	var levels []bitvector

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
		h.levelSizes = append(h.levelSizes, size)

		keys = redo
		redo = redo[:0] // tricky, sharing space with `keys`
		size = uint32(gamma * float64(len(keys)))
		size = (size + 63) &^ 63
		A.reset()
		collide.reset()
		level++
	}

	// Flatten levels into a single bitvector
	var totalWords uint32
	for _, lv := range levels {
		totalWords += uint32(len(lv))
	}
	h.b = make(bitvector, totalWords)
	var currentWord uint32
	for _, lv := range levels {
		copy(h.b[currentWord:], lv)
		currentWord += uint32(len(lv))
	}

	h.computeRanks()

	return &h
}

func (h *H) computeRanks() {
	var pop uint32
	// Pre-allocate ranks: 1 rank for every 8 uint64 words
	h.ranks = make([]uint32, 0, (len(h.b)+7)/8)
	for i, v := range h.b {
		if i%8 == 0 {
			h.ranks = append(h.ranks, pop)
		}
		pop += uint32(bits.OnesCount64(v))
	}
}

// Query returns the index of the key
func (h *H) Query(k tbits.BitString) uint64 {
	hash := k.Hash()
	h1, h2 := uint32(hash), uint32(hash>>32)

	var offset uint32
	for level, size := range h.levelSizes {
		idx := (h1 ^ bits.RotateLeft32(h2, level)) % size
		n := offset + (idx / 64)

		if h.b[n]&(1<<(idx%64)) == 0 {
			offset += (size + 63) / 64
			continue
		}

		rank := h.ranks[n/8]
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
	return len(h.b)*8 + len(h.ranks)*4 + len(h.levelSizes)*4
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
