package zft

import (
	"Thesis/bits"
	"math"
	"math/rand"
	"strings"
	"testing"
)

func TestStatistics_GetExitNode(t *testing.T) {
	t.Parallel()
	t.Run("CallCountIncrement", func(t *testing.T) {
		t.Parallel()
		zt := NewZFastTrie[int](-1)
		key := bits.NewFromText("test_key")
		zt.InsertBitString(key, 1)

		// Reset stats to have a clean state
		zt.stat = statistics{}

		const n = 50
		for i := 0; i < n; i++ {
			zt.GetExitNode(key)
		}

		if zt.stat.GetExitNodeCnt != n {
			t.Errorf("Expected GetExitNodeCnt to be %d, got %d", n, zt.stat.GetExitNodeCnt)
		}
	})

	t.Run("InnerLoopLogarithmicGrowth", func(t *testing.T) {
		t.Parallel()
		sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
		for range 10_000 {
			sizes = append(sizes, rand.Intn(4095)+1)
		}

		for _, s := range sizes {
			zt := NewZFastTrie[int](-1)

			key := buildBitString(s)

			expectedLog := math.Log2(float64(s)*8) + 3
			upperBound := expectedLog + 1.0

			zt.stat = statistics{}
			zt.GetExitNode(key)
			loops := zt.stat.GetExitNodeInnerLoopCnt

			if float64(loops) > upperBound {
				t.Errorf("Size %d: loops=%d. Expected approx %.2f (logarithmic). Growth check failed.", s, loops, expectedLog)
			}
		}
	})
}

func buildBitString(lenBytes int) bits.BitString {
	sb := strings.Builder{}
	for i := 0; i < lenBytes; i++ {
		sb.WriteByte('x')
	}
	keyStr := sb.String()
	key := bits.NewFromText(keyStr)
	return key
}
