//go:build amd64 && !noasm

package turboquant

func dequantizeBlock4_AVX2(data []byte, vec []float32, scale, zero float32, start, end int) int {
	blockLen := end - start
	if blockLen < 32 {
		return dequantizeBlock4_LUT(data, vec, scale, zero, start, end)
	}

	processed := 0
	i := 0
	dataOffset := 0

	for i+32 <= blockLen {
		dequantizeBlock4_AVX2_32(
			data[dataOffset:],
			vec[start+i:],
			scale,
			zero,
		)
		i += 32
		dataOffset += 16
		processed += 16
	}

	remaining := dequantizeBlock4_LUT(data[dataOffset:], vec[start+i:], scale, zero, 0, blockLen-i)
	processed += remaining

	return processed
}

func dequantizeBlock8_AVX2(data []byte, vec []float32, scale, zero float32, start, end int) int {
	return dequantizeBlock8_LUT(data, vec, scale, zero, start, end)
}

//go:noescape
func dequantizeBlock4_AVX2_32(data []byte, vec []float32, scale, zero float32)

//go:noescape
func dequantizeBlock8_AVX2_32(data []byte, vec []float32, scale, zero float32)
