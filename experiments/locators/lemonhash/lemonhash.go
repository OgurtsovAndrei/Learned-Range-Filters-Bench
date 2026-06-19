package lemonhash

/*
#cgo CXXFLAGS: -std=c++20
#cgo LDFLAGS: -L${SRCDIR}/build -llemonhash_wrapper -L${SRCDIR}/build/ext/LeMonHash -lsdsl -L${SRCDIR}/build/ext/LeMonHash/extlib/simpleRibbon -lSimpleRibbon -lRibbonSorter -L${SRCDIR}/build/ext/LeMonHash/extlib/tlx/tlx -ltlx -lc++ -lm
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"Thesis/bits"
	"Thesis/utils/errutil"
	mathbits "math/bits"
	"runtime"
	"unsafe"
)

type LeMonHash struct {
	ptr C.LeMonHashPtr
}

// reverseBitsInBytes reverses the bits within each byte of the slice in-place.
// This makes byte-wise lexicographical comparison (memcmp) equivalent to BitString.TrieCompare.
func reverseBitsInBytes(data []byte) {
	for i := 0; i < len(data); i++ {
		data[i] = mathbits.Reverse8(data[i])
	}
}

// New creates a new LeMonHash wrapper.
// The keys MUST be sorted and unique, just like in other MMPH implementations.
func New(keys []bits.BitString) *LeMonHash {
	if len(keys) == 0 {
		return &LeMonHash{ptr: nil}
	}
	if len(keys) == 1 {
		return &LeMonHash{ptr: nil} // We can return 0 for any query when N=1
	}

	numKeys := len(keys)

	// Prepare C arrays
	cStrings := make([]*C.char, numKeys)
	lengths := make([]C.size_t, numKeys)

	for i, k := range keys {
		data := k.Data()
		reverseBitsInBytes(data)
		if len(data) == 0 {
			cStrings[i] = nil
		} else {
			cStrings[i] = (*C.char)(C.CBytes(data))
		}
		lengths[i] = C.size_t(len(data))
	}

	ptr := C.lemonhash_vl_new((**C.char)(&cStrings[0]), (*C.size_t)(&lengths[0]), C.size_t(numKeys))

	// Free C strings
	for _, cStr := range cStrings {
		if cStr != nil {
			C.free(unsafe.Pointer(cStr))
		}
	}

	lh := &LeMonHash{ptr: ptr}
	runtime.SetFinalizer(lh, func(obj *LeMonHash) {
		if obj.ptr != nil {
			C.lemonhash_vl_free(obj.ptr)
			obj.ptr = nil
		}
	})

	return lh
}

func (lh *LeMonHash) Rank(key bits.BitString) int {
	if lh.ptr == nil {
		return 0
	}
	errutil.BugOn(key.Size() > 32*8, "Only keys up to 256 bits are supported")

	var buf [32]byte
	data := key.AppendToBytes(buf[:0])
	reverseBitsInBytes(data)
	var ptr *C.char
	if len(data) > 0 {
		ptr = (*C.char)(unsafe.Pointer(&data[0]))
	}

	rank := C.lemonhash_vl_query(lh.ptr, ptr, C.size_t(len(data)))
	return int(rank)
}

// rankRaw is a version of Rank that doesn't use Data() (which allocates).
// It's meant for benchmarking the CGO overhead itself.
func (lh *LeMonHash) rankRaw(data []byte) int {
	if lh.ptr == nil {
		return 0
	}
	// Note: for benchmarking rankRaw we assume data is already bit-reversed if needed,
	// or we just don't care because we only benchmark latency.
	var ptr *C.char
	if len(data) > 0 {
		ptr = (*C.char)(unsafe.Pointer(&data[0]))
	}
	rank := C.lemonhash_vl_query(lh.ptr, ptr, C.size_t(len(data)))
	return int(rank)
}

func (lh *LeMonHash) RankBatch(keys []bits.BitString, results []int) {
	if lh.ptr == nil || len(keys) == 0 {
		return
	}

	numKeys := len(keys)
	cKeys := make([]*C.char, numKeys)
	lengths := make([]C.size_t, numKeys)
	cResults := make([]C.uint64_t, numKeys)

	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(&cKeys[0])
	pinner.Pin(&lengths[0])
	pinner.Pin(&cResults[0])

	for i, k := range keys {
		data := k.Data()
		reverseBitsInBytes(data)
		if len(data) > 0 {
			pinner.Pin(&data[0])
			cKeys[i] = (*C.char)(unsafe.Pointer(&data[0]))
		}
		lengths[i] = C.size_t(len(data))
	}

	C.lemonhash_vl_query_batch(lh.ptr, (**C.char)(unsafe.Pointer(&cKeys[0])), (*C.size_t)(unsafe.Pointer(&lengths[0])), C.size_t(numKeys), (*C.uint64_t)(unsafe.Pointer(&cResults[0])))

	for i, r := range cResults {
		results[i] = int(r)
	}
}

// RankPair queries two keys in a single CGO call.
// Since arguments are passed directly, Go performs implicit pinning,
// avoiding the overhead of runtime.Pinner used in RankBatch.
func (lh *LeMonHash) RankPair(k1, k2 bits.BitString) (int, int) {
	if lh.ptr == nil {
		return 0, 0
	}
	errutil.BugOn(k1.Size() > 32*8 || k2.Size() > 32*8, "Only keys up to 256 bits are supported")

	var buf1, buf2 [32]byte
	d1 := k1.AppendToBytes(buf1[:0])
	d2 := k2.AppendToBytes(buf2[:0])
	reverseBitsInBytes(d1)
	reverseBitsInBytes(d2)

	var p1, p2 *C.char
	if len(d1) > 0 {
		p1 = (*C.char)(unsafe.Pointer(&d1[0]))
	}
	if len(d2) > 0 {
		p2 = (*C.char)(unsafe.Pointer(&d2[0]))
	}

	var r1, r2 C.uint64_t
	C.lemonhash_vl_query_pair(lh.ptr, p1, C.size_t(len(d1)), p2, C.size_t(len(d2)), &r1, &r2)

	return int(r1), int(r2)
}

// rankPairRaw is a zero-alloc version of RankPair for benchmarking.
func (lh *LeMonHash) rankPairRaw(d1, d2 []byte) (int, int) {
	if lh.ptr == nil {
		return 0, 0
	}

	var p1, p2 *C.char
	if len(d1) > 0 {
		p1 = (*C.char)(unsafe.Pointer(&d1[0]))
	}
	if len(d2) > 0 {
		p2 = (*C.char)(unsafe.Pointer(&d2[0]))
	}

	var r1, r2 C.uint64_t
	C.lemonhash_vl_query_pair(lh.ptr, p1, C.size_t(len(d1)), p2, C.size_t(len(d2)), &r1, &r2)

	return int(r1), int(r2)
}

func (lh *LeMonHash) ByteSize() int {
	if lh.ptr == nil {
		return 0
	}
	return int(C.lemonhash_vl_space_bits(lh.ptr) / 8)
}
