package turboquant

// dequantizeBlock8_LUT dequantizes an 8-bit block with SIMD-style loop unrolling.
func dequantizeBlock8_LUT(data []byte, vec []float32, scale, zero float32, start, end int) int {
	i := start
	offset := 0
	for ; i+8 <= end; i += 8 {
		vec[i+0] = float32(data[offset+0])*scale + zero
		vec[i+1] = float32(data[offset+1])*scale + zero
		vec[i+2] = float32(data[offset+2])*scale + zero
		vec[i+3] = float32(data[offset+3])*scale + zero
		vec[i+4] = float32(data[offset+4])*scale + zero
		vec[i+5] = float32(data[offset+5])*scale + zero
		vec[i+6] = float32(data[offset+6])*scale + zero
		vec[i+7] = float32(data[offset+7])*scale + zero
		offset += 8
	}
	for ; i < end; i++ {
		vec[i] = float32(data[offset])*scale + zero
		offset++
	}
	return offset
}

// dequantizeBlock4_LUT dequantizes a 4-bit block with SIMD-style loop unrolling.
func dequantizeBlock4_LUT(data []byte, vec []float32, scale, zero float32, start, end int) int {
	offset := 0
	i := start
	for ; i+4 <= end; i += 4 {
		packed0 := data[offset]
		packed1 := data[offset+1]

		vec[i+0] = float32(packed0>>4)*scale + zero
		vec[i+1] = float32(packed0&0x0F)*scale + zero
		vec[i+2] = float32(packed1>>4)*scale + zero
		vec[i+3] = float32(packed1&0x0F)*scale + zero
		offset += 2
	}
	for ; i < end; i += 2 {
		packed := data[offset]
		vec[i] = float32(packed>>4)*scale + zero
		if i+1 < end {
			vec[i+1] = float32(packed&0x0F)*scale + zero
		}
		offset++
	}
	return offset
}

// quantizeBlock8_LUT quantizes an 8-bit block with SIMD-style loop unrolling.
func quantizeBlock8_LUT(vec []float32, out []byte, scale, zero float32, start, end int) int {
	invScale := 1.0 / scale

	i := start
	j := 0
	for ; i+4 <= end; i += 4 {
		v0 := (vec[i+0] - zero) * invScale
		v1 := (vec[i+1] - zero) * invScale
		v2 := (vec[i+2] - zero) * invScale
		v3 := (vec[i+3] - zero) * invScale

		out[j+0] = fastRoundUint8(v0)
		out[j+1] = fastRoundUint8(v1)
		out[j+2] = fastRoundUint8(v2)
		out[j+3] = fastRoundUint8(v3)
		j += 4
	}
	for ; i < end; i++ {
		v := (vec[i] - zero) * invScale
		out[j] = fastRoundUint8(v)
		j++
	}
	return j
}

// fastRoundUint8 rounds a float32 to [0, 255] range efficiently.
func fastRoundUint8(v float32) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(int32(v + 0.5))
}

// quantizeBlock4_LUT quantizes a 4-bit block with SIMD-style loop unrolling.
func quantizeBlock4_LUT(vec []float32, out []byte, scale, zero float32, start, end int) int {
	invScale := 1.0 / scale

	i := start
	j := 0
	for ; i+2 <= end; i += 2 {
		v0 := (vec[i+0] - zero) * invScale
		v1 := (vec[i+1] - zero) * invScale

		q0 := fastRoundUint4(v0)
		q1 := fastRoundUint4(v1)

		out[j] = q0<<4 | q1
		j++
	}
	if i < end {
		v := (vec[i] - zero) * invScale
		out[j] = fastRoundUint4(v) << 4
		j++
	}
	return j
}

// fastRoundUint4 rounds a float32 to [0, 15] range efficiently.
func fastRoundUint4(v float32) byte {
	if v < 0 {
		return 0
	}
	if v > 15 {
		return 15
	}
	return byte(int32(v + 0.5))
}
