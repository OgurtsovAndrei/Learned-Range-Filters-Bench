package zft

import (
	"fmt"
	"testing"
)

func testTrie(t *testing.T, texts []string, isInverse bool) {
	tree := NewZFastTrie[bool](false)

	for i := 0; i < len(texts); i++ {
		tree.Insert(texts[i], true)
	}

	for i := 0; i < len(texts); i++ {
		if !tree.Contains(texts[i]) {
			t.Fatalf("Failed to find inserted text: %s\n\n%s\n\n%q", texts[i], tree.String(), texts)
		}
	}

	for i := 0; i < len(texts); i++ {
		for j := 0; j <= len(texts[i]); j++ { // Include empty prefix (j=0) and full string (j=len)
			prefix := texts[i][:j]
			if !tree.ContainsPrefix(prefix) {
				t.Fatalf("Failed to find prefix: %q (from %s)", prefix, texts[i])
			}
		}
	}

	for i := 0; i < len(texts); i++ {
		for j := 0; j < len(texts[i]); j++ {
			prefix := texts[i][:j] + "X"
			if tree.Contains(prefix) {
				t.Fatalf("Found non-existent text: %s", prefix)
			}
			if tree.ContainsPrefix(prefix) {
				t.Fatalf("Found non-existent prefix: %s", prefix)
			}
		}
	}

	fmt.Println(tree.String())

	for i := 0; i < len(texts); i++ {
		index := i
		if isInverse {
			index = len(texts) - 1 - i
		}
		tree.Erase(texts[index])
		if tree.Contains(texts[index]) {
			fmt.Println(tree.String())
			fmt.Println(tree.Contains(texts[index]))
			t.Fatalf("Found erased text: %s", texts[index])
		}
	}
}

// Ported from ZFastTrie/test.cpp
func TestTrie(t *testing.T) {
	t.Parallel()
	texts1 := []string{"A", "AA", "AAA", "AAAA", "AAAAA", "AAAAAA", "AAAAAAA", "AAAAAAAA", "B"}
	t.Run("A_B", func(t *testing.T) {
		t.Parallel()
		testTrie(t, texts1, false)
		testTrie(t, texts1, true)
	})

	texts2 := []string{"0000", "0000000100000011", "00000002", "0000000100000012", "000000010000001100000021"}
	t.Run("Complex_0", func(t *testing.T) {
		t.Parallel()
		testTrie(t, texts2, false)
		testTrie(t, texts2, true)
	})

	texts3 := []string{"0000000100000011", "00000001", "000000010000", "00000"}
	t.Run("Prefixes", func(t *testing.T) {
		t.Parallel()
		testTrie(t, texts3, false)
		testTrie(t, texts3, true)
	})

	texts4 := []string{"0a", "0b", "0c", "0d", "0e", "0f", "0"}
	t.Run("Siblings", func(t *testing.T) {
		t.Parallel()
		testTrie(t, texts4, false)
		testTrie(t, texts4, true)
	})

	texts5 := []string{"aa", "a", "b", "ba"}
	t.Run("Simple", func(t *testing.T) {
		t.Parallel()
		testTrie(t, texts5, false)
		testTrie(t, texts5, true)
	})
}
