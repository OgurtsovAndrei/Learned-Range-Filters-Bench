// Package original is the original BBHash implementation
package original

import (
	"math/bits"
	tbits "Thesis/bits"
)

type H struct {
	b     []bitvector
	ranks [][]uint64
}

func New(gamma float64, keys []tbits.BitString) *H {
	var h H
	var level uint32
	size := uint32(gamma * float64(len(keys)))
	size = (size + 63) &^ 63

	A := newbv(size)
	collide := newbv(size)
	var redo []tbits.BitString

	for len(keys) > 0 {
		for _, v := range keys {
			hash := v.Hash()
			h1, h2 := uint32(hash), uint32(hash>>32)
			idx := (h1 ^ bits.RotateLeft32(h2, int(level))) % size

			if collide.get(idx) == 1 { continue }
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
		h.b = append(h.b, bv)
		keys = redo
		redo = redo[:0]
		size = uint32(gamma * float64(len(keys)))
		size = (size + 63) &^ 63
		A.reset()
		collide.reset()
		level++
	}
	h.computeRanks()
	return &h
}

func (h *H) computeRanks() {
	var pop uint64
	for _, bv := range h.b {
		r := make([]uint64, 0, 1+(len(bv)/8))
		for i, v := range bv {
			if i%8 == 0 { r = append(r, pop) }
			pop += uint64(bits.OnesCount64(v))
		}
		h.ranks = append(h.ranks, r)
	}
}

func (h *H) Query(k tbits.BitString) uint64 {
	hash := k.Hash()
	h1, h2 := uint32(hash), uint32(hash>>32)
	for i, bv := range h.b {
		idx := (h1 ^ bits.RotateLeft32(h2, i)) % uint32(len(bv)*64)
		if bv.get(idx) == 0 { continue }
		rank := h.ranks[i][idx/512]
		for j := (idx / 64) &^ 7; j < idx/64; j++ {
			rank += uint64(bits.OnesCount64(bv[j]))
		}
		w := bv[idx/64]
		rank += uint64(bits.OnesCount64(w << (64 - (idx % 64))))
		return rank + 1
	}
	return 0
}

func (h *H) Size() int {
	var size int
	for _, v := range h.b { size += len(v) * 8 }
	for _, v := range h.ranks { size += len(v) * 8 }
	return size
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
	for i := range b { b[i] = 0 }
}
