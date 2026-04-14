package turboquant

// dotProdBlock8 returns sumQQ, sumQA, sumQB for 8-bit blocks.
func dotProdBlock8(a, b []byte) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock8_impl(a, b)
}

// dotProdBlock4 returns sumQQ, sumQA, sumQB for 4-bit blocks.
func dotProdBlock4(a, b []byte, blockLen int) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock4_unrolled(a, b, blockLen)
}
