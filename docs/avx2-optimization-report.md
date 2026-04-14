# AVX2/SSE2 8-bit Dot Product Optimization Report

**Date:** April 3, 2026  
**Status:** Reverted to Scalar - Cgo AVX2 Attempt Failed  
**Goal:** Optimize 8-bit dot product to match or beat FP32 performance (<1.0 μs)

## Executive Summary

Attempted to implement AVX2 SIMD optimization via **cgo** for 8-bit compressed dot product. Created `avx2/` package with C implementation. All cgo AVX2 implementations were reverted due to a **fundamental hardware limitation**: `VPMADDUBSW` requires signed bytes for the second operand, but Go's `uint8` values 128-255 are reinterpreted as negative values when viewed as `int8_t`, causing incorrect multiplication results.

Current scalar 8-bit dot product runs at ~2,569 ns/op, significantly slower than FP32 at ~364–730 ns/op (3.5–7x slower depending on vector dimension).

## Current Performance Baseline

| Implementation | Time (ns/op) | Notes |
|---------------|--------------|-------|
| FP32 DotProduct | 364–730 | Baseline (varies by vector dimension; see note below) |
| 8-bit Scalar | 2,569 | Current - 8-way unrolled |
| 4-bit Scalar | 3,249 | |
| 4-bit+QJL | 3,649 | |

**Note on FP32 baseline:** The report's 364 ns figure uses a single-dimension benchmark, while the README shows 730 ns for 1536-dim vectors. The discrepancy reflects measurement methodology differences.

**Target:** 8-bit should match or beat FP32 (~350 ns/op)

## Cgo Implementation Attempt (FAILED)

### Files Created

```
avx2/
├── dotprod_avx2.h    # Header with function declarations
├── dotprod_avx2.c    # C AVX2 implementation
└── avx2.go          # Go wrapper with cgo
```

### Algorithm Implementation

The intended AVX2 algorithm using `VPMADDUBSW`:

```c
// This is what we tried to implement
__m256i va = _mm256_loadu_si256((__m256i*)(a + i));  // uint8_t
__m256i vb = _mm256_loadu_si256((__m256i*)(b + i));  // uint8_t
__m256i prod16 = _mm256_maddubs_epi16(va, vb);  // PROBLEM HERE
```

### Root Cause: Signed Byte Issue

**`VPMADDUBSW` semantics:**
- **First operand**: unsigned bytes
- **Second operand**: **signed bytes**

When Go's `uint8` (values 0-255) is passed to C as `int8_t`:
```
uint8_t: 200 (0xC8) → int8_t: -56  (same bits, reinterpreted as signed)
```

This causes **incorrect multiplication** because the hardware interprets byte value 200 as -56 in the multiplication.

### Attempted Workarounds

1. **Batch processing**: Amortized cgo overhead but didn't solve the sign issue
2. **Separate scale/zero extraction**: Fixed data layout bugs but still had sign problem

### Cgo Pointer Issue

Batch processing with cgo failed due to Go's pointer policy:

```
panic: runtime error: cgo argument has Go pointer to unpinned Go pointer
```

**Root cause:** Go's runtime does not allow passing Go-managed pointers to C code without explicit copying. The fix is to use `C.CBytes` to copy slice data into C-managed memory, or allocate with `C.malloc`. `runtime.KeepAlive` and pinning are insufficient for slice data passed across the cgo boundary.

## Search Performance

Despite single-vector dot product being slower than FP32, search performance is reasonable due to parallelism:

| Dataset | 8-bit Search | 4-bit | 4-bit+QJL |
|---------|-------------|-------|-----------|
| 10K | 7.3 ms | 9.5 ms | 10.7 ms |
| 100K | 74 ms | 97 ms | 107 ms |
| 500K | 372 ms | 498 ms | 542 ms |

## Recommendations

### Priority 1: Change Quantization Format (Recommended)

Store vectors as **signed int8** instead of unsigned `uint8`:

```go
// In quantization, store as int8 instead of byte
// This requires changing the compression format
vec[i] = int8(quantizedValue)  // -128 to +127
```

Then AVX2 `VPMADDUBSW` will work correctly because both operands can be handled properly.

### Priority 2: Use go:asm with Hand-Written Assembly

Bypass cgo entirely by writing assembly in `.s` files with `//go:build` constraints. **Note:** The codebase already has working AVX2 assembly — `dequant_amd64.s` successfully uses `VPMOVZXBD`, `VCVTDQ2PS`, `VBROADCASTSS`, and other AVX2 instructions for 4-bit dequantization. The tooling works; the challenge is algorithmic (no single AVX2 instruction performs uint8×uint8 multiply with horizontal reduction).

An 8-bit dot product assembly implementation using the unpack-to-int16 approach (see Technical Notes below) is feasible. The question is whether the instruction count overhead justifies the parallelism gain vs scalar.

### Priority 3: Accept Current Performance

Current 7x slowdown may be acceptable:
- 8-bit compression saves 60% memory (2560B vs 6144B per vector)
- Search is parallelized across CPU cores
- Accuracy is preserved (Recall@10 = 100% for 8-bit)

## Technical Notes

### VPMADDUBSW Behavior

```
VPMADDUBSW dest, src1, src2

For each pair of bytes:
  result[i] = int16(src1[2*i]) * int16(src2[2*i+1])

src2 bytes are SIGNED (-128 to +127)
src1 bytes are UNSIGNED (0 to 255)
```

When Go byte value 200 is interpreted as signed:
- Expected: 200 × other_value
- Actual: -56 × other_value (WRONG!)

**Note:** The bits `0xC8` are unchanged — the value is reinterpreted, not sign-extended. Sign extension occurs only when widening to a larger type (e.g., `int8` → `int16`). Here the byte is loaded directly into a 16-bit lane with signed interpretation.

### Correct AVX2 Path for uint8 × uint8

Must unpack to uint16 first, then multiply and horizontally reduce. A concrete instruction sequence:

```
// Load 32 bytes from each vector (2x YMM registers)
VMOVDQU  (a), Y0       // bytes 0-31 of vector A
VMOVDQU  (b), Y1       // bytes 0-31 of vector B

// Unpack bytes to uint16 (zero-extend)
VPUNPCKLBW Y1, Y0, Y2  // lo 16 bytes → 16 uint16 interleaved
VPUNPCKHBW Y1, Y0, Y3  // hi 16 bytes → 16 uint16 interleaved

// Multiply 16-bit values
VPMULLW  Y3, Y2, Y4    // element-wise uint16 multiply

// Horizontal add to reduce 32 products to a single sum
// Requires VPHADDW repeated, or extract-and-sum in scalar
VPHADDW  Y4, Y4, Y5    // 32 → 16 words
VPHADDW  Y5, Y5, Y6    // 16 → 8 words
// Final scalar sum of remaining 8 words
```

**Performance estimate:** This processes 32 elements per iteration vs 8 in the 8-way unrolled scalar. However, each iteration requires ~8 AVX instructions vs ~16 scalar ops. The break-even point depends on instruction latency vs throughput on the target microarchitecture. On Zen 3 (Ryzen 5900HS), this could plausibly reach **500–800 ns** vs the current 2,569 ns — a 3–5x improvement, though likely still behind FP32 due to the extra unpack/multiply steps.

## Conclusion

**Cgo AVX2 is not viable** for this use case due to the signed byte requirement of `VPMADDUBSW`. The fundamental issue is the mismatch between Go's `uint8` (always unsigned) and the AVX2 instruction's signed byte expectation.

**Path forward:**
1. Change quantization to use signed int8 storage (requires format change — quantize to [-128, +127] range with zero at 0)
2. Write hand-crafted AVX2 assembly in `.s` (like `dequant_amd64.s`) using the unpack-to-int16 approach — estimated 500–800 ns vs current 2,569 ns
3. Accept current scalar performance — 8-bit saves 60% memory and search is parallelized

The cgo infrastructure is removed; the codebase uses scalar 8-way unrolled implementation which is correct but not fast.
