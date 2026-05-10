package keygen

import (
	"math/rand"
)

// LoadKeysForSpec is a helper for benchmarks to load keys based on distribution name.
func LoadKeysForSpec(distName string, n int) ([]uint64, error) {
	switch distName {
	case "sosd_fb":
		return LoadSOSDUint64(SOSDPath("fb_200M_uint64"), n)
	case "uniform":
		rng := rand.New(rand.NewSource(0xBEEF))
		ks := make([]uint64, n)
		for i := range ks {
			ks[i] = rng.Uint64()
		}
		return ks, nil
	default:
		return nil, nil
	}
}

