//go:build !amd64
// +build !amd64

package turboquant

// dotProdBlock8_SIMD is the generic fallback for dotProdBlock8.
func dotProdBlock8_SIMD(a, b []byte) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock8Generic(a, b)
}

// dotProdBlock4_SIMD is the generic fallback for dotProdBlock4.
func dotProdBlock4_SIMD(a, b []byte, blockLen int) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock4Generic(a, b, blockLen)
}
