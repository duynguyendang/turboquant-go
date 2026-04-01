//go:build !amd64

package turboquant

func dequantizeBlock4_AVX2(data []byte, vec []float32, scale, zero float32, start, end int) int {
	return dequantizeBlock4_LUT(data, vec, scale, zero, start, end)
}

func dequantizeBlock8_AVX2(data []byte, vec []float32, scale, zero float32, start, end int) int {
	return dequantizeBlock8_LUT(data, vec, scale, zero, start, end)
}
