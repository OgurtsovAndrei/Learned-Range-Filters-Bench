// Package boomphf is a fast perfect hash function for massive key sets
// code taken from https://github.com/dgryski/go-boomphf
/*
   https://arxiv.org/abs/1702.03154
*/
package inline_uint64

import (
	"math/bits"
)

// H is the optimized BBHash implementation using inlined level sizes and 8-word alignment
type H struct {
	b     []uint64
	ranks []uint32
}

// Gamma is the recommended default value for controlling space vs. construction speed
const Gamma = 2.0

// NewDefault constructs a perfect hash function with the default gamma value (2.0)
func NewDefault(keys []uint64) *H {
	return New(2.0, keys)
}

// New constructs a perfect hash function for the keys. The gamma value controls the space used.
func New(gamma float64, keys []uint64) *H {
	var h H

	var level uint32
	size := uint32(gamma * float64(len(keys)))
	size = (size + 63) &^ 63

	A := newbv(size)
	collide := newbv(size)

	var redo []uint64
	var levels [][]uint64
	var levelSizes []uint32

	for len(keys) > 0 {
		for _, v := range keys {
			hash := xorshiftMult64(v)
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
			hash := xorshiftMult64(v)
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
		size = (size + 63) &^ 63
		A.reset()
		collide.reset()
		level++
	}

	// Flatten with inlined sizes and 8-word alignment
	var totalWords uint32
	for _, lv := range levels {
		totalWords += 8 + uint32(len(lv))
	}
	h.b = make([]uint64, totalWords)
	var current uint32
	for i, lv := range levels {
		h.b[current] = uint64(levelSizes[i])
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

		for i := 0; i < 8; i++ {
			if (offset+uint32(i))%8 == 0 {
				h.ranks = append(h.ranks, pop)
			}
		}

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
func (h *H) Query(k uint64) uint64 {
	hash := xorshiftMult64(k)
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
		for j := n &^ 7; j < n; j++ {
			rank += uint32(bits.OnesCount64(h.b[j]))
		}
		rank += uint32(bits.OnesCount64(h.b[n] << (64 - (idx % 64))))

		return uint64(rank + 1)
	}

	return 0
}

func (h *H) Size() int {
	return len(h.b)*8 + len(h.ranks)*4
}

func (h *H) ByteSize() int {
	return h.Size()
}

// 64-bit xorshift multiply rng from http://vigna.di.unimi.it/ftp/papers/xorshift.pdf
func xorshiftMult64(x uint64) uint64 {
	x ^= x >> 12 // a
	x ^= x << 25 // b
	x ^= x >> 27 // c
	return x * 2685821657736338717
}

type bitvector []uint64

func newbv(size uint32) bitvector { return make([]uint64, uint(size+63)/64) }
func (b bitvector) get(bit uint32) uint {
	shift := bit % 64
	bb := b[bit/64]
	bb &= (1 << shift)
	return uint(bb >> shift)
}
func (b bitvector) set(bit uint32) { b[bit/64] |= (1 << (bit % 64)) }
func (b bitvector) reset() {
	for i := range b {
		b[i] = 0
	}
}
