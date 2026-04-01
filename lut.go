package turboquant

// LUT (Look-Up Table) for quantization and dequantization
// Provides fast byte → float32 and float32 → byte conversions

// DequantLUT8 is a pre-computed look-up table for 8-bit dequantization.
// For each byte value 0-255, stores the float32 value.
var DequantLUT8 [256]float32

// QuantLUT8 is a look-up table for 8-bit quantization rounding.
// Maps float32 v (expected to be in [0, 255]) to the nearest byte.
var QuantLUT8 [256]float32

// QuantLUT8Int is the integer version for fast clamping.
var QuantLUT8Int [256]int32

// DequantLUT4 is a look-up table for 4-bit dequantization.
var DequantLUT4 [16]float32

// QuantLUT4 is a look-up table for 4-bit quantization rounding.
var QuantLUT4 [16]float32

func init() {
	// Initialize 8-bit dequantization LUT: val = byte * scale + zero
	// We store the raw byte value, multiply by scale and add zero at runtime
	for i := 0; i < 256; i++ {
		DequantLUT8[i] = float32(i)
	}

	// Initialize 8-bit quantization LUT
	// Maps float32 v to nearest byte with clamping to [0, 255]
	// We store v directly and do round-to-nearest at runtime
	for i := 0; i < 256; i++ {
		QuantLUT8[i] = float32(i)
		QuantLUT8Int[i] = int32(i)
	}

	// Initialize 4-bit dequantization LUT
	for i := 0; i < 16; i++ {
		DequantLUT4[i] = float32(i)
	}

	// Initialize 4-bit quantization LUT
	for i := 0; i < 16; i++ {
		QuantLUT4[i] = float32(i)
	}
}

// dequantizeBlock8_LUT dequantizes an 8-bit block using LUT with SIMD-style unrolling.
func dequantizeBlock8_LUT(data []byte, vec []float32, scale, zero float32, start, end int) int {
	// Process 8 bytes at a time with loop unrolling
	i := start
	offset := 0
	for ; i+8 <= end; i += 8 {
		// Unroll 8 iterations
		vec[i+0] = DequantLUT8[data[offset+0]]*scale + zero
		vec[i+1] = DequantLUT8[data[offset+1]]*scale + zero
		vec[i+2] = DequantLUT8[data[offset+2]]*scale + zero
		vec[i+3] = DequantLUT8[data[offset+3]]*scale + zero
		vec[i+4] = DequantLUT8[data[offset+4]]*scale + zero
		vec[i+5] = DequantLUT8[data[offset+5]]*scale + zero
		vec[i+6] = DequantLUT8[data[offset+6]]*scale + zero
		vec[i+7] = DequantLUT8[data[offset+7]]*scale + zero
		offset += 8
	}
	// Handle remainder
	for ; i < end; i++ {
		vec[i] = DequantLUT8[data[offset]]*scale + zero
		offset++
	}
	return offset
}

// dequantizeBlock8_LUT_FMA dequantizes using FMA-style computation.
func dequantizeBlock8_LUT_FMA(data []byte, vec []float32, scale, zero float32, start, end int) int {
	// Fused multiply-add: vec[i] = DequantLUT8[byte] * scale + zero
	// Modern CPUs can fuse this into single instruction
	i := start
	offset := 0
	for ; i+4 <= end; i += 4 {
		vec[i+0] = DequantLUT8[data[offset+0]]*scale + zero
		vec[i+1] = DequantLUT8[data[offset+1]]*scale + zero
		vec[i+2] = DequantLUT8[data[offset+2]]*scale + zero
		vec[i+3] = DequantLUT8[data[offset+3]]*scale + zero
		offset += 4
	}
	for ; i < end; i++ {
		vec[i] = DequantLUT8[data[offset]]*scale + zero
		offset++
	}
	return offset
}

// dequantizeBlock4_LUT dequantizes a 4-bit block using LUT with SIMD-style unrolling.
func dequantizeBlock4_LUT(data []byte, vec []float32, scale, zero float32, start, end int) int {
	offset := 0
	i := start
	for ; i+4 <= end; i += 4 {
		// Process 4 pairs of nibbles (2 bytes) at a time
		packed0 := data[offset]
		packed1 := data[offset+1]

		vec[i+0] = DequantLUT4[packed0>>4]*scale + zero
		vec[i+1] = DequantLUT4[packed0&0x0F]*scale + zero
		vec[i+2] = DequantLUT4[packed1>>4]*scale + zero
		vec[i+3] = DequantLUT4[packed1&0x0F]*scale + zero
		offset += 2
	}
	// Handle remainder
	for ; i < end; i += 2 {
		packed := data[offset]
		vec[i] = DequantLUT4[packed>>4]*scale + zero
		if i+1 < end {
			vec[i+1] = DequantLUT4[packed&0x0F]*scale + zero
		}
		offset++
	}
	return offset
}

// quantizeBlock8_LUT quantizes an 8-bit block using LUT with SIMD-style unrolling.
func quantizeBlock8_LUT(vec []float32, out []byte, scale, zero float32, start, end int) int {
	// Compute (v - zero) / scale for each element, then round to nearest byte
	invScale := 1.0 / scale

	i := start
	j := 0
	for ; i+4 <= end; i += 4 {
		// Unroll 4 iterations
		v0 := (vec[i+0] - zero) * invScale
		v1 := (vec[i+1] - zero) * invScale
		v2 := (vec[i+2] - zero) * invScale
		v3 := (vec[i+3] - zero) * invScale

		// Fast round to nearest with clamping using bit operations
		out[j+0] = fastRoundUint8(v0)
		out[j+1] = fastRoundUint8(v1)
		out[j+2] = fastRoundUint8(v2)
		out[j+3] = fastRoundUint8(v3)
		j += 4
	}
	// Handle remainder
	for ; i < end; i++ {
		v := (vec[i] - zero) * invScale
		out[j] = fastRoundUint8(v)
		j++
	}
	return j
}

// fastRoundUint8 rounds a float32 to [0, 255] range efficiently.
func fastRoundUint8(v float32) byte {
	// Add 0.5 and truncate for rounding
	// Clamp to [0, 255] using bit operations
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(int32(v + 0.5))
}

// quantizeBlock4_LUT quantizes a 4-bit block using LUT with SIMD-style unrolling.
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
	// Handle remainder
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
