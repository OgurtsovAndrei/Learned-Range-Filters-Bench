package hzft

import (
	"Thesis/bits"
	"Thesis-bench-industry/experiments/trie/zft"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHZFastTrie_GetExistingPrefix(t *testing.T) {
	t.Parallel()
	// Строки из Figure 1 в статье [cite: 112, 113]
	s0 := bits.NewFromBinary("001001010")
	s1 := bits.NewFromBinary("0010011010010")
	s2 := bits.NewFromBinary("00100110101")
	keys := []bits.BitString{s0, s1, s2}

	hzft := NewHZFastTrie[uint32](keys)
	require.NotNil(t, hzft)

	// Build reference trie separately for comparison
	referenceTrie, _ := zft.BuildFromIterator(bits.NewSliceBitStringIterator(keys))

	// Согласно Figure 2, корень имеет расширение длины 6 (001001) [cite: 153, 154]
	// Любой префикс короче или равный 6 битам имеет узлом выхода корень [cite: 102]
	// Узел alpha имеет имя длины 7 (0010011) [cite: 113, 153]
	tests := []struct {
		pattern  string
		expected int64
	}{
		{"0010010", 7},        // Узел alpha (длина имени 7)
		{"0010011", 7},        // Узел alpha
		{"001001101", 7},      // Все еще в пределах ребра узла alpha
		{"0010011010010", 11}, // Лист (длина имени 11)
		{"00100110101", 11},   // Другой лист (длина имени 11)
		{"0010", 0},           // Узел выхода — корень (имя пустой строки)
		{"", 0},               // Корень
	}

	for _, tt := range tests {
		t.Run(tt.pattern, func(t *testing.T) {
			p := bits.NewFromBinary(tt.pattern)
			result := hzft.GetExistingPrefix(p)

			// Проверка основного результата
			require.Equal(t, tt.expected, result)

			// Сверка с эталонным узлом выхода из ZFastTrie [cite: 176, 224]
			expectedNode := referenceTrie.GetExitNode(p)
			var refExpected int64
			if expectedNode != referenceTrie.Root {
				// Длина имени узла в сжатом боре [cite: 101, 237]
				refExpected = int64(expectedNode.NameLength)
			}
			require.Equal(t, refExpected, result)
		})
	}
}

func TestHZFastTrie_Empty(t *testing.T) {
	t.Parallel()
	var keys []bits.BitString
	hzft := NewHZFastTrie[uint32](keys)
	if hzft != nil {
		t.Error("Expected nil for empty keys")
	}
}

func TestHZFastTrie_Simple2(t *testing.T) {
	t.Parallel()
	// Строки из Figure 1 в статье [cite: 112, 113]
	s0 := bits.NewFromBinary("00")
	s1 := bits.NewFromBinary("11")
	keys := []bits.BitString{s0, s1}

	hzft := NewHZFastTrie[uint32](keys)
	require.NotNil(t, hzft)

	// Build reference trie separately for comparison
	referenceTrie, _ := zft.BuildFromIterator(bits.NewSliceBitStringIterator(keys))

	tests := []struct {
		pattern  string
		expected int64
	}{
		{"0", 1},
	}

	for _, tt := range tests {
		t.Run(tt.pattern, func(t *testing.T) {
			p := bits.NewFromBinary(tt.pattern)
			result := hzft.GetExistingPrefix(p)

			// Проверка основного результата
			require.Equal(t, tt.expected, result)

			// Сверка с эталонным узлом выхода из ZFastTrie [cite: 176, 224]
			expectedNode := referenceTrie.GetExitNode(p)
			var refExpected int64
			if expectedNode != referenceTrie.Root {
				// Длина имени узла в сжатом боре [cite: 101, 237]
				refExpected = int64(expectedNode.NameLength)
			}
			require.Equal(t, refExpected, result)
		})
	}
}

func TestHZFastTrie_CornerCases(t *testing.T) {
	t.Parallel()
	t.Run("SingleKey", func(t *testing.T) {
		t.Parallel()
		key := bits.NewFromBinary("010101")
		hzft := NewHZFastTrie[uint32]([]bits.BitString{key})
		require.NotNil(t, hzft)

		require.Equal(t, int64(0), hzft.GetExistingPrefix(bits.NewFromBinary("010")))
		require.Equal(t, int64(0), hzft.GetExistingPrefix(bits.NewFromBinary("010101")))
	})

	t.Run("DivergentAtFirstBit", func(t *testing.T) {
		t.Parallel()
		s0 := bits.NewFromBinary("011")
		s1 := bits.NewFromBinary("100")
		hzft := NewHZFastTrie[uint32]([]bits.BitString{s0, s1})

		require.Equal(t, int64(0), hzft.GetExistingPrefix(bits.NewFromBinary("")))
		require.Equal(t, int64(1), hzft.GetExistingPrefix(bits.NewFromBinary("0")))
		require.Equal(t, int64(1), hzft.GetExistingPrefix(bits.NewFromBinary("1")))
		require.Equal(t, int64(1), hzft.GetExistingPrefix(bits.NewFromBinary("01")))
	})

	t.Run("LongCommonPrefix", func(t *testing.T) {
		t.Parallel()
		common := "0101010101010101"
		s0 := bits.NewFromBinary(common + "00")
		s1 := bits.NewFromBinary(common + "11")
		hzft := NewHZFastTrie[uint32]([]bits.BitString{s0, s1})

		require.Equal(t, int64(0), hzft.GetExistingPrefix(bits.NewFromBinary("0101")))
		require.Equal(t, int64(0), hzft.GetExistingPrefix(bits.NewFromBinary(common)))
		require.Equal(t, int64(17), hzft.GetExistingPrefix(bits.NewFromBinary(common+"0")))
	})

	t.Run("DeeplyNestedTrie", func(t *testing.T) {
		t.Parallel()
		keys := []bits.BitString{
			bits.NewFromBinary("0000"),
			bits.NewFromBinary("0001"),
			bits.NewFromBinary("0011"),
			bits.NewFromBinary("0111"),
		}
		hzft := NewHZFastTrie[uint32](keys)

		fmt.Println(hzft)

		require.Equal(t, int64(0), hzft.GetExistingPrefix(bits.NewFromBinary("0")))
		require.Equal(t, int64(3), hzft.GetExistingPrefix(bits.NewFromBinary("001")))
		require.Equal(t, int64(4), hzft.GetExistingPrefix(bits.NewFromBinary("0001")))
	})

	t.Run("DeeplyNestedTrie1", func(t *testing.T) {
		t.Parallel()
		keys := []bits.BitString{
			bits.NewFromBinary("1000"),
			bits.NewFromBinary("1100"),
			bits.NewFromBinary("1110"),
			bits.NewFromBinary("1111"),
		}
		hzft := NewHZFastTrie[uint32](keys)

		require.Equal(t, int64(0), hzft.GetExistingPrefix(bits.NewFromBinary("1")))
		require.Equal(t, int64(3), hzft.GetExistingPrefix(bits.NewFromBinary("111")))
		require.Equal(t, int64(4), hzft.GetExistingPrefix(bits.NewFromBinary("1111")))
	})
}
