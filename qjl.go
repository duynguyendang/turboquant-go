package turboquant

import (
	"math"
	"math/bits"
)

func f32tof16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := (b >> 16) & 0x8000
	exp := int32((b>>23)&0xFF) - 127 + 15
	mantissa := (b >> 13) & 0x3FF
	if exp <= 0 {
		return uint16(sign)
	}
	if exp >= 31 {
		return uint16(sign | 0x7C00)
	}
	return uint16(sign | (uint32(exp) << 10) | mantissa)
}

func f16tof32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mantissa := uint32(h&0x03FF) << 13
	if exp == 0 {
		return math.Float32frombits(sign | mantissa)
	}
	if exp == 31 {
		return math.Float32frombits(sign | 0x7F800000 | mantissa)
	}
	return math.Float32frombits(sign | ((exp - 15 + 127) << 23) | mantissa)
}

func qjlBitCount64(paddedDim int) int {
	return (paddedDim + 63) / 64
}

func qjlBitSize64(paddedDim int) int {
	return qjlBitCount64(paddedDim) * 8
}

func packQJLBits64(residual []float32, bits []uint64) {
	for i := range bits {
		var word uint64
		base := i * 64
		limit := base + 64
		if limit > len(residual) {
			limit = len(residual)
		}
		for j := base; j < limit; j++ {
			if residual[j] >= 0 {
				word |= 1 << uint(j-base)
			}
		}
		bits[i] = word
	}
}

func qjlCorrectionPerBlock(bitsA, bitsB []uint64, numBlocks, blockSize int) []int64 {
	corr := make([]int64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		qjlIdx := b / 2
		shift := (b % 2) * 32
		xor := (bitsA[qjlIdx] ^ bitsB[qjlIdx]) >> shift
		pop := bits.OnesCount32(uint32(xor))
		corr[b] = int64(blockSize) - 2*int64(pop)
	}
	return corr
}

func computeQJLWeightPerBlock(padded []float32, dequantized []float32, numBlocks, blockSize int) []uint16 {
	weights := make([]uint16, numBlocks)
	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(padded) {
			end = len(padded)
		}
		var sumAbs float32
		for i := start; i < end; i++ {
			r := padded[i] - dequantized[i]
			if r < 0 {
				sumAbs -= r
			} else {
				sumAbs += r
			}
		}
		w := sumAbs / float32(end-start)
		weights[b] = f32tof16(w)
	}
	return weights
}
