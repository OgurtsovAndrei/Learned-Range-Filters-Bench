package zft

import (
	"encoding/json"
	"os"
	"testing"
)

type replayTestOperation struct {
	Op       string `json:"op"`
	Key      string `json:"key"`
	Value    bool   `json:"value,omitempty"`
	Expected bool   `json:"expected,omitempty"`
}

type replayTestHistory struct {
	Seed       int64                 `json:"seed"`
	Operations []replayTestOperation `json:"operations"`
}

func TestTrie_ReplayFromHistory(t *testing.T) {
	t.Parallel()
	filePath := "out/fail_history_1762347109872759000.json"

	data, err := os.ReadFile(filePath)
	if err != nil {
		t.Skipf("fail to read file %s", filePath)
		//t.Fatalf("Failed to read history file %q: %v", filePath, err)
	}

	var history replayTestHistory
	if err := json.Unmarshal(data, &history); err != nil {
		t.Fatalf("Failed to unmarshal history file %q: %v", filePath, err)
	}

	t.Logf("--- Replaying test from file: %s (Seed: %d) ---", filePath, history.Seed)

	tree := NewZFastTrie[bool](false)
	groundTruth := make(map[string]bool)
	prefixGroundTruth := make(map[string]bool)

	for i, op := range history.Operations {
		opNum := i + 1

		switch op.Op {
		case "Insert":
			if op.Key == "SKIPPED" {
				continue
			}
			if op.Value {
				tree.Insert(op.Key, true)
				groundTruth[op.Key] = true
				for j := 0; j <= len(op.Key); j++ {
					prefixGroundTruth[op.Key[:j]] = true
				}
				if !tree.Contains(op.Key) {
					t.Fatalf("Replay op %d [Insert(%q)]: Failed to find just-inserted key", opNum, op.Key)
				}
			}

		case "Erase":
			if op.Key == "SKIPPED" {
				continue
			}
			tree.Erase(op.Key)
			delete(groundTruth, op.Key)

			if tree.Contains(op.Key) {
				t.Fatalf("Replay op %d [Erase(%q)]: Found just-deleted key", opNum, op.Key)
			}

		case "Contains":
			actual := tree.Contains(op.Key)
			expectedGT, _ := groundTruth[op.Key]

			if actual != op.Expected {
				t.Fatalf("Replay op %d [Contains(%q)]: Mismatch. Expected from history: %v, Got: %v", opNum, op.Key, op.Expected, actual)
			}
			if actual != expectedGT {
				t.Fatalf("Replay op %d [Contains(%q)]: Mismatch. Expected from replay ground truth: %v, Got: %v", opNum, op.Key, expectedGT, actual)
			}

		case "ContainsPrefix":
			actual := tree.ContainsPrefix(op.Key)
			expectedGT, _ := prefixGroundTruth[op.Key]

			if op.Expected && !expectedGT {
				t.Logf("Replay op %d [ContainsPrefix(%q)]: Warning: history expected true, but replay ground truth does not have prefix. This might be ok if replay logic differs.", opNum, op.Key)
			}

			if !op.Expected && expectedGT {
				t.Fatalf("Replay op %d [ContainsPrefix(%q)]: Mismatch. History expected false, but replay ground truth has prefix.", opNum, op.Key)
			}

			if actual != op.Expected {
				t.Fatalf("Replay op %d [ContainsPrefix(%q)]: Mismatch. Expected from history: %v, Got: %v", opNum, op.Key, op.Expected, actual)
			}

		default:
			t.Fatalf("Replay op %d: Unknown operation %q in history file", opNum, op.Op)
		}
	}

	t.Logf("--- Successfully replayed %d operations from %s ---", len(history.Operations), filePath)

	t.Log("--- Final Replay State Validation ---")
	for key := range groundTruth {
		if !tree.Contains(key) {
			t.Fatalf("Final replay check failed: key %q in ground truth but not in Trie", key)
		}
	}
	if int(tree.size) != len(groundTruth) {
		t.Fatalf("Final replay size mismatch. Expected: %d, Got: %d", len(groundTruth), tree.size)
	}
	t.Log("--- Final Replay State OK ---")
}
