//go:build amd64 && !noasm

package turboquant

import "math/bits"

func qjlCorrectionSIMD(bitsA, bitsB []uint64) int64 {
	var total int64
	for i := 0; i < len(bitsA); i++ {
		total += int64(bits.OnesCount64(bitsA[i] ^ bitsB[i]))
	}
	return int64(len(bitsA)*64) - 2*total
}
