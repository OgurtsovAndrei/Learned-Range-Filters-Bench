package zft

import (
	"Thesis/bits"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"
)

type testOperation struct {
	Op       string `json:"op"`
	Key      string `json:"key"`
	Size     uint32 `json:"size"`
	Value    any    `json:"value,omitempty"`
	Expected any    `json:"expected,omitempty"`
}

type testHistory struct {
	Seed       int64           `json:"seed"`
	Operations []testOperation `json:"operations"`
}

type mapKey struct {
	data string
	size uint32
}

func toMapKey(bs bits.BitString) mapKey {
	return mapKey{
		data: string(bs.Data()),
		size: bs.Size(),
	}
}

func saveHistoryAndFail(t *testing.T, seed int64, history []testOperation, format string, args ...interface{}) {
	t.Helper()

	if err := os.MkdirAll("out", 0755); err != nil {
		t.Logf("!!! Failed to create out directory: %v", err)
	}

	fileName := fmt.Sprintf("out/fail_history_%d.json", seed)
	historyData := testHistory{
		Seed:       seed,
		Operations: history,
	}

	jsonData, err := json.MarshalIndent(historyData, "", "  ")
	if err != nil {
		t.Logf("!!! Failed to marshal failure history: %v", err)
	} else {
		if err := os.WriteFile(fileName, jsonData, 0644); err != nil {
			t.Logf("!!! Failed to write failure history to %s: %v", fileName, err)
		} else {
			t.Logf("--- FAILURE HISTORY SAVED TO: %s ---", fileName)
		}
	}

	t.Fatalf(format, args...)
}

func TestTrie_HeavyRandom_BitString_Ops(t *testing.T) {
	t.Parallel()
	for test_id := range 100 {
		fmt.Println("TestTrie_HeavyRandom_BitString_Ops iteration:", test_id)
		seed := time.Now().UnixNano()
		r := rand.New(rand.NewSource(seed))

		tree := NewZFastTrie[bool](false)

		// Ground truth uses the composite key (Data + Size)
		groundTruth := make(map[mapKey]bool)
		insertedKeys := make([]bits.BitString, 0)

		numOperations := 10_000
		maxBitLen := 128

		history := make([]testOperation, 0, numOperations)

		defer func() {
			if r := recover(); r != nil {
				saveHistoryAndFail(t, seed, history, "Test panicked: %v", r)
			}
		}()

		for i := 0; i < numOperations; i++ {
			op := r.Intn(100)
			genLen := r.Intn(maxBitLen) + 1

			if op < 45 {
				s := GenerateBitString(genLen, r)
				mk := toMapKey(s)

				history = append(history, testOperation{Op: "InsertBitString", Key: s.PrettyString(), Size: s.Size(), Value: true})

				if _, exists := groundTruth[mk]; !exists {
					groundTruth[mk] = true
					insertedKeys = append(insertedKeys, s)
					tree.InsertBitString(s, true)

					if !tree.ContainsBitString(s) {
						saveHistoryAndFail(t, seed, history, "Failed to find just-inserted key: %s", s.PrettyString())
					}
				}
			} else if op < 80 {
				if len(insertedKeys) == 0 {
					history = append(history, testOperation{Op: "EraseBitString", Key: "SKIPPED_EMPTY"})
					continue
				}
				idx := r.Intn(len(insertedKeys))
				s := insertedKeys[idx]
				mk := toMapKey(s)

				history = append(history, testOperation{Op: "EraseBitString", Key: s.PrettyString(), Size: s.Size()})

				delete(groundTruth, mk)

				insertedKeys[idx] = insertedKeys[len(insertedKeys)-1]
				insertedKeys = insertedKeys[:len(insertedKeys)-1]

				tree.EraseBitString(s)

				if tree.ContainsBitString(s) {
					saveHistoryAndFail(t, seed, history, "Found just-deleted key: %s", s.PrettyString())
				}
			} else {
				s := GenerateBitString(genLen, r)
				mk := toMapKey(s)
				expected, _ := groundTruth[mk]

				history = append(history, testOperation{Op: "ContainsBitString", Key: s.PrettyString(), Size: s.Size(), Expected: expected})

				actual := tree.ContainsBitString(s)

				if actual != expected {
					saveHistoryAndFail(t, seed, history, "ContainsBitString mismatch for key %s. Expected: %v, Got: %v", s.PrettyString(), expected, actual)
				}
			}
		}

		for _, key := range insertedKeys {
			if !tree.ContainsBitString(key) {
				saveHistoryAndFail(t, seed, history, "Final check failed: key %s in ground truth but not in Trie", key.PrettyString())
			}
		}

		if int(tree.size) != len(groundTruth) {
			saveHistoryAndFail(t, seed, history, "Final size mismatch. Expected: %d, Got: %d", len(groundTruth), tree.size)
		}
	}
}

func TestTrie_HeavyRandom_BitString_Get(t *testing.T) {
	t.Parallel()
	for test_id := range 100 {
		fmt.Println("TestTrie_HeavyRandom_BitString_Get iteration:", test_id)
		seed := time.Now().UnixNano()
		r := rand.New(rand.NewSource(seed))

		emptyValue := -1
		tree := NewZFastTrie[int](emptyValue)

		groundTruth := make(map[mapKey]int)
		insertedKeys := make([]bits.BitString, 0)

		numOperations := 100_000
		maxBitLen := 128

		history := make([]testOperation, 0, numOperations)

		defer func() {
			if r := recover(); r != nil {
				saveHistoryAndFail(t, seed, history, "Test panicked: %v", r)
			}
		}()

		for i := 0; i < numOperations; i++ {
			op := r.Intn(100)
			genLen := r.Intn(maxBitLen) + 1

			if op < 50 {
				s := GenerateBitString(genLen, r)
				mk := toMapKey(s)
				v := r.Intn(1000000)

				history = append(history, testOperation{Op: "InsertBitString", Key: s.PrettyString(), Size: s.Size(), Value: v})

				if _, exists := groundTruth[mk]; !exists {
					insertedKeys = append(insertedKeys, s)
				}
				groundTruth[mk] = v
				tree.InsertBitString(s, v)
			} else {
				var s bits.BitString
				if len(insertedKeys) > 0 && r.Float64() < 0.7 {
					s = insertedKeys[r.Intn(len(insertedKeys))]
				} else {
					s = GenerateBitString(genLen, r)
				}
				mk := toMapKey(s)

				expected, ok := groundTruth[mk]
				if !ok {
					expected = emptyValue
				}

				history = append(history, testOperation{Op: "GetBitString", Key: s.PrettyString(), Size: s.Size(), Expected: expected})

				actual := tree.GetBitString(s)

				if actual != expected {
					saveHistoryAndFail(t, seed, history, "GetBitString mismatch for key %s. Expected: %v, Got: %v", s.PrettyString(), expected, actual)
				}
			}
		}

		for _, bs := range insertedKeys {
			mk := toMapKey(bs)
			expectedValue := groundTruth[mk]
			actualValue := tree.GetBitString(bs)
			if actualValue != expectedValue {
				saveHistoryAndFail(t, seed, history, "Final check GetBitString mismatch for key %s. Expected: %v, Got: %v", bs.PrettyString(), expectedValue, actualValue)
			}
		}
	}
}
