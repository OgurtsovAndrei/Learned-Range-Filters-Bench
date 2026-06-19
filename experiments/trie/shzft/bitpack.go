package shzft

// packBits takes an array of values and packs them into a dense []uint64 slice,
// where each value uses exactly 'bitWidth' bits.
func packBits(values []uint64, bitWidth int) []uint64 {
	if len(values) == 0 {
		return nil
	}
	if bitWidth == 0 {
		return []uint64{} // Should not happen for delta > 0, but safe
	}

	totalBits := len(values) * bitWidth
	numWords := (totalBits + 63) / 64
	packed := make([]uint64, numWords)

	for i, val := range values {
		bitPos := i * bitWidth
		wordIdx := bitPos / 64
		bitOffset := uint(bitPos % 64)

		// Mask value to ensure it doesn't overflow its width
		mask := uint64(1<<bitWidth) - 1
		maskedVal := val & mask

		// Write to the first word
		packed[wordIdx] |= maskedVal << bitOffset

		// Handle overflow into the next word
		bitsAvailableInWord := 64 - int(bitOffset)
		if bitsAvailableInWord < bitWidth {
			// Write the remaining bits to the next word
			bitsWritten := bitsAvailableInWord
			packed[wordIdx+1] |= maskedVal >> uint(bitsWritten)
		}
	}

	return packed
}

// unpackBit extracts a value of 'bitWidth' bits from the packed []uint64 slice
// at the given 'index' (which is the i-th value, not bit position).
func unpackBit(packed []uint64, index int, bitWidth int) uint64 {
	if bitWidth == 0 {
		return 0
	}

	bitPos := index * bitWidth
	wordIdx := bitPos / 64
	bitOffset := uint(bitPos % 64)

	// Read from the first word
	val := packed[wordIdx] >> bitOffset

	// Check if value spans across two words
	bitsAvailableInWord := 64 - int(bitOffset)
	if bitsAvailableInWord < bitWidth {
		// Read the remaining bits from the next word
		bitsRead := bitsAvailableInWord
		nextWordVal := packed[wordIdx+1]
		val |= nextWordVal << uint(bitsRead)
	}

	// Mask out the extraneous higher bits
	mask := uint64(1<<bitWidth) - 1
	return val & mask
}
