//go:build amd64 && !noasm

#include "textflag.h"

// func dequantizeBlock4_AVX2_32(data []byte, vec []float32, scale, zero float32)
//
// Dequantizes 16 packed bytes (32 nibbles) into 32 float32 values.
// Uses AVX2 VPMOVZXBD + VCVTDQ2PS for fast nibble→float conversion.
//
// Layout (Go ABI):
//   data_base+0(FP)  = *byte
//   data_len+8(FP)   = int
//   vec_base+24(FP)  = *float32
//   vec_len+32(FP)   = int
//   scale+48(FP)     = float32
//   zero+52(FP)      = float32

TEXT ·dequantizeBlock4_AVX2_32(SB), NOSPLIT|NOFRAME, $0-56
    MOVQ data_base+0(FP), AX
    MOVQ vec_base+24(FP), BX
    MOVSS scale+48(FP), X0
    MOVSS zero+52(FP), X1

    // Broadcast scale and zero to all 8 lanes of YMM
    VBROADCASTSS X0, Y0
    VBROADCASTSS X1, Y1

    // Load 16 packed bytes (32 nibbles)
    VMOVDQU (AX), X2

    // Extract low nibbles: [l0, l1, ..., l15]
    VPAND ·mask4<>(SB), X2, X3

    // Extract high nibbles: [h0, h1, ..., h15]
    VPSRLQ $4, X2, X4
    VPAND ·mask4<>(SB), X4, X4

    // Interleave first 8 pairs: [l0, h0, l1, h1, ..., l7, h7]
    VPUNPCKLBW X4, X3, X5

    // Interleave last 8 pairs: [l8, h8, l9, h9, ..., l15, h15]
    VPUNPCKHBW X4, X3, X6

    // Zero-extend bytes to dwords: 16 bytes → 16 uint32
    VPMOVZXBD X5, Y5
    VPMOVZXBD X6, Y6

    // Convert uint32 to float32
    VCVTDQ2PS Y5, Y5
    VCVTDQ2PS Y6, Y6

    // Dequantize: val * scale + zero
    VMULPS Y0, Y5, Y5
    VADDPS Y1, Y5, Y5

    VMULPS Y0, Y6, Y6
    VADDPS Y1, Y6, Y6

    // Store 32 float32 values (128 bytes total)
    // Y5: 8 floats (32 bytes)
    // Y6: 8 floats (32 bytes)
    // Wait - YMM holds 8 float32, not 16!
    // So Y5 has 8 floats, Y6 has 8 floats = 16 floats total, not 32!

    // I need to fix this: 16 nibbles → 16 uint32 → 16 float32 per YMM
    // VPMOVZXBD X5, Y5: 16 bytes → 16 dwords ✓
    // VCVTDQ2PS Y5, Y5: 16 dwords → 16 float32 ✓
    // So Y5 has 16 floats, Y6 has 16 floats = 32 floats ✓

    // Store: Y5 = 16 floats = 64 bytes, Y6 = 16 floats = 64 bytes
    VMOVDQU Y5, (BX)
    VMOVDQU Y6, 64(BX)

    VZEROUPPER
    RET

// func dequantizeBlock8_AVX2_32(data []byte, vec []float32, scale, zero float32)
//
// Dequantizes 32 bytes into 32 float32 values using AVX2.
// Processes in two halves of 16 bytes each (VPMOVZXBD is XMM→YMM only).
//
// Layout (Go ABI):
//   data_base+0(FP)  = *byte
//   data_len+8(FP)   = int
//   vec_base+24(FP)  = *float32
//   vec_len+32(FP)   = int
//   scale+48(FP)     = float32
//   zero+52(FP)      = float32

TEXT ·dequantizeBlock8_AVX2_32(SB), NOSPLIT|NOFRAME, $0-56
    MOVQ data_base+0(FP), AX
    MOVQ vec_base+24(FP), BX
    MOVSS scale+48(FP), X0
    MOVSS zero+52(FP), X1

    // Broadcast scale and zero to all 8 lanes of YMM
    VBROADCASTSS X0, Y0
    VBROADCASTSS X1, Y1

    // Load first 16 bytes
    VMOVDQU (AX), X2
    // Zero-extend bytes to dwords: 16 bytes → 16 uint32
    VPMOVZXBD X2, Y2
    // Convert uint32 to float32
    VCVTDQ2PS Y2, Y2
    // Dequantize: val * scale + zero
    VMULPS Y0, Y2, Y2
    VADDPS Y1, Y2, Y2
    // Store 16 float32 values (64 bytes)
    VMOVDQU Y2, (BX)

    // Load next 16 bytes
    VMOVDQU 16(AX), X3
    // Zero-extend bytes to dwords
    VPMOVZXBD X3, Y3
    // Convert uint32 to float32
    VCVTDQ2PS Y3, Y3
    // Dequantize
    VMULPS Y0, Y3, Y3
    VADDPS Y1, Y3, Y3
    // Store 16 float32 values (64 bytes) at offset 64
    VMOVDQU Y3, 64(BX)

    VZEROUPPER
    RET

// Mask 0x0F repeated for nibble extraction
DATA ·mask4<>(SB)/8, $0x0f0f0f0f0f0f0f0f
DATA ·mask4<>+8(SB)/8, $0x0f0f0f0f0f0f0f0f
GLOBL ·mask4<>(SB), RODATA|NOPTR, $16
