package turboquant

import (
	"math"
	"math/bits"
)

const (
	BitWidth4  = 4
	BitWidth8  = 8
	DefaultDim = 1536
)

// HybridConfig holds quantization parameters.
type HybridConfig struct {
	BitWidth  int
	BlockSize int
	UseLUT    bool
	EnableQJL bool
}

// DefaultHybridConfig returns a default hybrid configuration.
func DefaultHybridConfig() *HybridConfig {
	return &HybridConfig{
		BitWidth:  BitWidth8,
		BlockSize: 32,
		UseLUT:    true,
		EnableQJL: false,
	}
}

// HybridVectorSize returns the size in bytes of a hybrid-compressed vector.
func HybridVectorSize(dim int, cfg *HybridConfig) int {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}
	paddedDim := nextPow2(dim)
	numBlocks := (paddedDim + cfg.BlockSize - 1) / cfg.BlockSize
	baseSize := 0
	switch cfg.BitWidth {
	case BitWidth8:
		baseSize = numBlocks * (8 + cfg.BlockSize)
	case BitWidth4:
		baseSize = numBlocks * (8 + cfg.BlockSize/2)
	default:
		baseSize = numBlocks * (8 + cfg.BlockSize)
	}
	if cfg.EnableQJL && cfg.BitWidth == BitWidth4 {
		baseSize += numBlocks*2 + qjlBitSize64(paddedDim)
	}
	return baseSize
}

// QuantizeHybrid compresses a float32 vector using FWHT + block-wise quantization.
// Returns the compressed byte representation.
func QuantizeHybrid(vec []float32, cfg *HybridConfig) []byte {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	dim := len(vec)
	paddedDim := nextPow2(dim)

	padded := make([]float32, paddedDim)
	copy(padded, vec)

	FWHT_Optimized(padded)

	invNorm := 1.0 / float32(math.Sqrt(float64(paddedDim)))
	for i := range padded {
		padded[i] *= invNorm
	}

	blockSize := cfg.BlockSize
	numBlocks := (paddedDim + blockSize - 1) / blockSize
	outSize := HybridVectorSize(paddedDim, cfg)
	out := make([]byte, outSize)

	offset := 0
	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > paddedDim {
			end = paddedDim
		}

		minVal, maxVal := float32(math.MaxFloat32), float32(-math.MaxFloat32)
		for i := start; i < end; i++ {
			if padded[i] < minVal {
				minVal = padded[i]
			}
			if padded[i] > maxVal {
				maxVal = padded[i]
			}
		}

		var scale, zero float32
		if maxVal == minVal {
			scale = 1.0
			zero = minVal
		} else {
			scale = (maxVal - minVal) / float32(int(1)<<cfg.BitWidth-1)
			zero = minVal
		}

		putFloat32(out[offset:], scale)
		offset += 4
		putFloat32(out[offset:], zero)
		offset += 4

		switch cfg.BitWidth {
		case BitWidth8:
			if cfg.UseLUT {
				offset += quantizeBlock8_LUT(padded, out[offset:offset+blockSize], scale, zero, start, end)
			} else {
				for i := start; i < end; i++ {
					v := (padded[i] - zero) / scale
					q := int(math.Round(float64(v)))
					if q < 0 {
						q = 0
					} else if q > 255 {
						q = 255
					}
					out[offset] = byte(q)
					offset++
				}
			}
			for i := end - start; i < blockSize; i++ {
				out[offset] = 0
				offset++
			}
		case BitWidth4:
			if cfg.UseLUT {
				offset += quantizeBlock4_LUT(padded, out[offset:offset+blockSize/2], scale, zero, start, end)
			} else {
				for i := start; i < end; i += 2 {
					v1 := (padded[i] - zero) / scale
					q1 := int(math.Round(float64(v1)))
					if q1 < 0 {
						q1 = 0
					} else if q1 > 15 {
						q1 = 15
					}
					var q2 int
					if i+1 < end {
						v2 := (padded[i+1] - zero) / scale
						q2 = int(math.Round(float64(v2)))
						if q2 < 0 {
							q2 = 0
						} else if q2 > 15 {
							q2 = 15
						}
					}
					out[offset] = byte(q1<<4 | q2)
					offset++
				}
			}
			for i := (end - start + 1) / 2; i < blockSize/2; i++ {
				out[offset] = 0
				offset++
			}
		}
	}

	if cfg.EnableQJL && cfg.BitWidth == BitWidth4 {
		dequantized := DequantizeHybridNoFWHT(out[:offset], paddedDim, cfg)

		weights := computeQJLWeightPerBlock(padded, dequantized, numBlocks, blockSize)
		for b := 0; b < numBlocks; b++ {
			putUint16(out[offset+b*2:], weights[b])
		}
		offset += numBlocks * 2

		residual := make([]float32, paddedDim)
		for i := 0; i < dim; i++ {
			residual[i] = padded[i] - dequantized[i]
		}

		bitCount := qjlBitCount64(paddedDim)
		bits := make([]uint64, bitCount)
		packQJLBits64(residual, bits)
		for i := 0; i < bitCount; i++ {
			putUint64(out[offset+i*8:], bits[i])
		}
	}

	return out
}

// DequantizeHybridNoFWHT decompresses without the inverse FWHT step.
// Used internally for QJL residual computation.
func DequantizeHybridNoFWHT(data []byte, paddedDim int, cfg *HybridConfig) []float32 {
	vec := make([]float32, paddedDim)
	blockSize := cfg.BlockSize
	numBlocks := (paddedDim + blockSize - 1) / blockSize

	offset := 0
	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > paddedDim {
			end = paddedDim
		}

		scale := getFloat32(data[offset:])
		offset += 4
		zero := getFloat32(data[offset:])
		offset += 4

		switch cfg.BitWidth {
		case BitWidth8:
			for i := start; i < end; i++ {
				vec[i] = float32(data[offset])*scale + zero
				offset++
			}
			offset += blockSize - (end - start)
		case BitWidth4:
			for i := start; i < end; i += 2 {
				packed := data[offset]
				vec[i] = float32(packed>>4)*scale + zero
				if i+1 < end {
					vec[i+1] = float32(packed&0x0F)*scale + zero
				}
				offset++
			}
			offset += blockSize/2 - (end-start+1)/2
		}
	}
	return vec
}

// DequantizeHybrid decompresses a hybrid-compressed vector back to float32.
func DequantizeHybrid(data []byte, dim int, cfg *HybridConfig) []float32 {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	paddedDim := nextPow2(dim)
	blockSize := cfg.BlockSize
	numBlocks := (paddedDim + blockSize - 1) / blockSize

	qjlOffset := 0
	switch cfg.BitWidth {
	case BitWidth8:
		qjlOffset = numBlocks * (8 + cfg.BlockSize)
	case BitWidth4:
		qjlOffset = numBlocks * (8 + cfg.BlockSize/2)
		if cfg.EnableQJL {
			qjlOffset += numBlocks*2 + qjlBitSize64(paddedDim)
		}
	}

	vecData := data
	if cfg.EnableQJL && cfg.BitWidth == BitWidth4 && len(data) > qjlOffset {
		vecData = data[:numBlocks*(8+blockSize/2)]
	}

	vec := DequantizeHybridNoFWHT(vecData, paddedDim, cfg)

	FWHT_Optimized(vec)
	invNorm := 1.0 / float32(math.Sqrt(float64(paddedDim)))
	for i := range vec {
		vec[i] *= invNorm
	}

	result := make([]float32, dim)
	copy(result, vec[:dim])
	return result
}

// DotProductHybrid computes the dot product between two compressed vectors
// without fully dequantizing them. Returns the cosine similarity (since vectors
// are L2-normalized after FWHT).
func DotProductHybrid(a, b []byte, dim int, cfg *HybridConfig) float32 {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	paddedDim := nextPow2(dim)
	expectedSize := HybridVectorSize(paddedDim, cfg)
	if len(a) < expectedSize || len(b) < expectedSize {
		return 0
	}

	blockSize := cfg.BlockSize
	numBlocks := (paddedDim + blockSize - 1) / blockSize

	var totalSum float32
	offsetA := 0
	offsetB := 0

	bitWidth := cfg.BitWidth

	for block := 0; block < numBlocks; block++ {
		blockLen := blockSize
		if (block+1)*blockSize > paddedDim {
			blockLen = paddedDim - block*blockSize
		}

		scaleA := getFloat32(a[offsetA:])
		zeroA := getFloat32(a[offsetA+4:])
		scaleB := getFloat32(b[offsetB:])
		zeroB := getFloat32(b[offsetB+4:])

		offsetA += 8
		offsetB += 8

		var sumQQ, sumQA, sumQB int64

		if bitWidth == BitWidth8 {
			sumQQ, sumQA, sumQB = dotProdBlock8(a[offsetA:offsetA+blockLen], b[offsetB:offsetB+blockLen])
			offsetA += blockSize
			offsetB += blockSize
		} else {
			sumQQ, sumQA, sumQB = dotProdBlock4(a[offsetA:offsetA+blockLen/2], b[offsetB:offsetB+blockLen/2], blockLen)
			offsetA += blockSize / 2
			offsetB += blockSize / 2
		}

		totalSum += scaleA*scaleB*float32(sumQQ) +
			scaleA*zeroB*float32(sumQA) +
			scaleB*zeroA*float32(sumQB) +
			float32(blockLen)*zeroA*zeroB
	}

	if cfg.EnableQJL && bitWidth == BitWidth4 {
		baseSize := numBlocks * (8 + blockSize/2)
		weightOff := baseSize

		var correctionSum float32
		for block := 0; block < numBlocks; block++ {
			wA := f16tof32(getUint16(a[weightOff+block*2:]))
			wB := f16tof32(getUint16(b[weightOff+block*2:]))

			qjlIdx := block / 2
			shift := (block % 2) * 32
			bitCount := qjlBitCount64(paddedDim)
			if qjlIdx < bitCount {
				xor := (getUint64(a[baseSize+numBlocks*2+qjlIdx*8:]) ^ getUint64(b[baseSize+numBlocks*2+qjlIdx*8:])) >> shift
				pop := bitsOnesCount32(uint32(xor))
				corr := int64(blockSize) - 2*int64(pop)
				correctionSum += wA * wB * float32(corr)
			}
		}
		totalSum += correctionSum
	}

	return totalSum
}

func bitsOnesCount32(v uint32) int {
	return bits.OnesCount32(v)
}

// DotProductHybridFull dequantizes both vectors and computes full dot product.
// This is slower but useful for verification.
func DotProductHybridFull(a, b []byte, dim int, cfg *HybridConfig) float32 {
	vecA := DequantizeHybrid(a, dim, cfg)
	vecB := DequantizeHybrid(b, dim, cfg)
	return DotProduct(vecA, vecB)
}
