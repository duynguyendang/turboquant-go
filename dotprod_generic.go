//go:build !amd64 || noasm

package turboquant

// dotProdBlock8_impl is the generic fallback for dotProdBlock8.
func dotProdBlock8_impl(a, b []byte) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock8Generic(a, b)
}

// dotProdBlock4_unrolled is the generic fallback for dotProdBlock4.
func dotProdBlock4_unrolled(a, b []byte, blockLen int) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock4Generic(a, b, blockLen)
}

// dotProdBlock8Generic computes sumQQ, sumQA, sumQB for 8-bit blocks
// with values stored as signed int8.
func dotProdBlock8Generic(a, b []byte) (sumQQ, sumQA, sumQB int64) {
	for i := 0; i < len(a); i++ {
		qa := int64(int8(a[i]))
		qb := int64(int8(b[i]))
		sumQQ += qa * qb
		sumQA += qa
		sumQB += qb
	}
	return
}

// dotProdBlock4Generic computes sumQQ, sumQA, sumQB for 4-bit blocks.
func dotProdBlock4Generic(a, b []byte, blockLen int) (sumQQ, sumQA, sumQB int64) {
	for i := 0; i < blockLen; i++ {
		byteA := a[i/2]
		byteB := b[i/2]
		var qa, qb int64
		if i%2 == 0 {
			qa = int64(byteA >> 4)
			qb = int64(byteB >> 4)
		} else {
			qa = int64(byteA & 0x0F)
			qb = int64(byteB & 0x0F)
		}
		sumQQ += qa * qb
		sumQA += qa
		sumQB += qb
	}
	return
}
