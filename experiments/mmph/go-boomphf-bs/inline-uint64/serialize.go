package inline_uint64

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
)

// The serialization format for the H structure is as follows (all values are LittleEndian):
//
// - uint32: Length of h.b (number of uint64 words)
// - h.b data: uint64 words
// - uint32: Length of h.ranks (number of uint32 values)
// - h.ranks data: uint32 values

func (h *H) Serialize() ([]byte, error) {
	var buf []byte

	// Serialize h.b
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(h.b)))
	for _, v := range h.b {
		buf = binary.LittleEndian.AppendUint64(buf, v)
	}

	// Serialize h.ranks
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(h.ranks)))
	for _, v := range h.ranks {
		buf = binary.LittleEndian.AppendUint32(buf, v)
	}

	return buf, nil
}

func Deserialize(data []byte, target *H) error {
	r := bytes.NewReader(data)

	// Read h.b
	var bLen uint32
	if err := binary.Read(r, binary.LittleEndian, &bLen); err != nil {
		return errors.New("failed to read b length")
	}

	target.b = make([]uint64, bLen)
	for i := uint32(0); i < bLen; i++ {
		var val uint64
		if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
			return errors.New("failed to read b data")
		}
		target.b[i] = val
	}

	// Read h.ranks
	var rankLen uint32
	if err := binary.Read(r, binary.LittleEndian, &rankLen); err != nil {
		if err == io.EOF {
			// Older version might not have ranks if it was built differently,
			// but here we expect them.
			return errors.New("failed to read ranks length")
		}
		return errors.New("failed to read ranks length")
	}

	target.ranks = make([]uint32, rankLen)
	for i := uint32(0); i < rankLen; i++ {
		var val uint32
		if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
			return errors.New("failed to read ranks data")
		}
		target.ranks[i] = val
	}

	if r.Len() != 0 {
		return errors.New("trailing data after successful deserialization")
	}

	return nil
}
