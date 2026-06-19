package zft

import (
	"Thesis/bits"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestTrie_BitString_Exhaustive8Bit(t *testing.T) {
	t.Parallel()
	numIterations := 1000
	for iter := 0; iter < numIterations; iter++ {
		fmt.Println("TestTrie_BitString_Exhaustive8Bit iteration:", iter)
		seed := time.Now().UnixNano()
		r := rand.New(rand.NewSource(seed))
		tree := NewZFastTrie[int](-1)

		allKeys := make([]string, 256)
		for i := 0; i < 256; i++ {
			allKeys[i] = string([]byte{byte(i)})
		}

		r.Shuffle(len(allKeys), func(i, j int) { allKeys[i], allKeys[j] = allKeys[j], allKeys[i] })

		keysToInsert := make(map[string]bool)
		for i := 0; i < 128; i++ {
			keysToInsert[allKeys[i]] = true
		}

		insertedSoFar := make(map[string]int)
		for _, key := range allKeys {
			insertedSoFar[key] = -1
		}

		for opNum, key := range allKeys {
			shouldInsert := keysToInsert[key]

			if shouldInsert {
				value := int(key[0])
				tree.Insert(key, value)
				insertedSoFar[key] = value
			}

			for _, checkKey := range allKeys {
				shouldBePresent := insertedSoFar[checkKey]
				actual := tree.GetBitString(bits.NewFromText(checkKey))

				if actual != shouldBePresent {
					t.Fatalf("Seed %d, Iter %d, Op %d (Key %v): Mismatch for checkKey %v. Expected: %v, Got: %v",
						seed, iter, opNum, []byte(key), []byte(checkKey), shouldBePresent, actual)
				}
			}
		}

		if int(tree.size) != len(keysToInsert) {
			t.Fatalf("Seed %d, Iter %d: Final size mismatch. Expected: %d, Got: %d",
				seed, iter, len(keysToInsert), tree.size)
		}
	}
}
