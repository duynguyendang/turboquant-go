//go:build amd64 && !noasm
// +build amd64,!noasm

package turboquant

func dotProdBlock8_AVX2(a, b []byte) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock8_scalar_impl(a, b)
}

func dotProdBlock4_SIMD(a, b []byte, blockLen int) (sumQQ, sumQA, sumQB int64) {
	var qq0, qq1 int64
	var qa0, qa1 int64
	var qb0, qb1 int64

	i := 0
	for ; i+2 <= blockLen; i += 2 {
		byteA := a[i/2]
		byteB := b[i/2]

		ha := int64(byteA >> 4)
		hb := int64(byteB >> 4)
		qq0 += ha * hb
		qa0 += ha
		qb0 += hb

		la := int64(byteA & 0x0F)
		lb := int64(byteB & 0x0F)
		qq1 += la * lb
		qa1 += la
		qb1 += lb
	}

	sumQQ = qq0 + qq1
	sumQA = qa0 + qa1
	sumQB = qb0 + qb1

	for ; i < blockLen; i++ {
		byteA := a[i/2]
		byteB := b[i/2]
		if i%2 == 0 {
			qa := int64(byteA >> 4)
			qb := int64(byteB >> 4)
			sumQQ += qa * qb
			sumQA += qa
			sumQB += qb
		} else {
			qa := int64(byteA & 0x0F)
			qb := int64(byteB & 0x0F)
			sumQQ += qa * qb
			sumQA += qa
			sumQB += qb
		}
	}

	return
}

func dotProdBlock8_SIMD(a, b []byte) (sumQQ, sumQA, sumQB int64) {
	return dotProdBlock8_AVX2(a, b)
}

func dotProdBlock8_scalar_impl(a, b []byte) (sumQQ, sumQA, sumQB int64) {
	n := len(a)

	var qq0, qq1, qq2, qq3 int64
	var qa0, qa1, qa2, qa3 int64
	var qb0, qb1, qb2, qb3 int64

	i := 0
	for ; i+8 <= n; i += 8 {
		a0, a1, a2, a3 := int64(a[i]), int64(a[i+1]), int64(a[i+2]), int64(a[i+3])
		a4, a5, a6, a7 := int64(a[i+4]), int64(a[i+5]), int64(a[i+6]), int64(a[i+7])
		b0, b1, b2, b3 := int64(b[i]), int64(b[i+1]), int64(b[i+2]), int64(b[i+3])
		b4, b5, b6, b7 := int64(b[i+4]), int64(b[i+5]), int64(b[i+6]), int64(b[i+7])

		qq0 += a0 * b0
		qq1 += a1 * b1
		qq2 += a2 * b2
		qq3 += a3 * b3
		qq0 += a4 * b4
		qq1 += a5 * b5
		qq2 += a6 * b6
		qq3 += a7 * b7

		qa0 += a0
		qa1 += a1
		qa2 += a2
		qa3 += a3
		qa0 += a4
		qa1 += a5
		qa2 += a6
		qa3 += a7

		qb0 += b0
		qb1 += b1
		qb2 += b2
		qb3 += b3
		qb0 += b4
		qb1 += b5
		qb2 += b6
		qb3 += b7
	}

	sumQQ = qq0 + qq1 + qq2 + qq3
	sumQA = qa0 + qa1 + qa2 + qa3
	sumQB = qb0 + qb1 + qb2 + qb3

	for ; i < n; i++ {
		qa := int64(a[i])
		qb := int64(b[i])
		sumQQ += qa * qb
		sumQA += qa
		sumQB += qb
	}

	return
}
