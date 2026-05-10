package datasets

import (
	"encoding/binary"
	"fmt"
	"os"
	"sort"
)

// SOSDKeyType selects the on-disk integer width for an SOSD binary file.
type SOSDKeyType int

const (
	SOSDUint64 SOSDKeyType = 64
	SOSDUint32 SOSDKeyType = 32
)

// SOSDReader reads a SOSD-format binary file:
//
//	[uint64 count, little-endian][count × {uint32|uint64}, little-endian]
//
// It returns sorted, deduplicated keys as []uint64.
type SOSDReader struct {
	Path     string
	Label    string
	KeyType  SOSDKeyType
	MaxKeys  int // if > 0, return at most MaxKeys (after sort+dedupe).
}

func (r *SOSDReader) Name() string { return r.Label }

func (r *SOSDReader) Keys() ([]uint64, error) {
	f, err := os.Open(r.Path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", r.Path, err)
	}
	defer f.Close()

	var count uint64
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read count from %s: %w", r.Path, err)
	}

	n := int(count)
	keys := make([]uint64, n)

	switch r.KeyType {
	case SOSDUint64:
		if err := binary.Read(f, binary.LittleEndian, keys); err != nil {
			return nil, fmt.Errorf("read uint64 keys from %s: %w", r.Path, err)
		}
	case SOSDUint32:
		raw := make([]uint32, n)
		if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
			return nil, fmt.Errorf("read uint32 keys from %s: %w", r.Path, err)
		}
		for i, v := range raw {
			keys[i] = uint64(v)
		}
	default:
		return nil, fmt.Errorf("SOSDReader %s: unsupported KeyType %d", r.Path, r.KeyType)
	}

	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	j := 0
	for i := 1; i < len(keys); i++ {
		if keys[i] != keys[j] {
			j++
			keys[j] = keys[i]
		}
	}
	keys = keys[:j+1]

	if r.MaxKeys > 0 && len(keys) > r.MaxKeys {
		keys = keys[:r.MaxKeys]
	}
	return keys, nil
}
