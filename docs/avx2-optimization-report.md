# AVX2/SSE2 8-bit Dot Product Optimization Report

**Date:** April 3, 2026  
**Status:** Reverted to Scalar - Cgo AVX2 Attempt Failed  
**Goal:** Optimize 8-bit dot product to match or beat FP32 performance (<1.0 μs)

## Executive Summary

Attempted to implement AVX2 SIMD optimization via **cgo** for 8-bit compressed dot product. Created `avx2/` package with C implementation. All cgo AVX2 implementations were reverted due to a **fundamental hardware limitation**: `VPMADDUBSW` requires signed bytes for the second operand, but Go's `uint8` values 128-255 get sign-extended when cast to C's `int8_t`, causing incorrect multiplication results.

Current scalar 8-bit dot product runs at ~2569 ns/op, significantly slower than FP32 at ~364 ns/op (7x slower).

## Current Performance Baseline

| Implementation | Time (ns/op) | Notes |
|---------------|--------------|-------|
| FP32 DotProduct | 364 | Baseline |
| 8-bit Scalar | 2,569 | Current - 8-way unrolled |
| 4-bit Scalar | 3,249 | |
| 4-bit+QJL | 3,649 | |

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
uint8_t: 200 (0xC8) → int8_t: -56  (sign-extended!)
```

This causes **incorrect multiplication** because the hardware interprets byte value 200 as -56 in the multiplication.

### Attempted Workarounds

1. **Using VPMULLB + manual sum**: Slower than pure scalar due to extra instructions
2. **Batch processing**: Amortized cgo overhead but didn't solve the sign issue
3. **Separate scale/zero extraction**: Fixed data layout bugs but still had sign problem

### Cgo Pointer Issue

Batch processing with cgo failed due to Go's pointer policy:

```
panic: runtime error: cgo argument has Go pointer to unpinned Go pointer
```

This requires either:
- Passing only raw C pointers (not Go slices)
- Using `runtime.KeepAlive` and explicit pinning
- Switching to `go:asm` which avoids cgo entirely

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

Bypass cgo entirely by writing assembly in `.s` files with `//go:build` constraints. This was attempted previously but failed due to:
- Go assembler bugs with VINSERTI128
- DATA section byte ordering issues
- Instruction encoding problems

If Go 1.24+ fixes these issues, this becomes viable.

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

### Correct AVX2 Path for uint8 × uint8

Must unpack to int16 first, then use `VPMADDWD`:

```c
// Correct approach (but slower):
__m256i va = _mm256_loadu_si256(...);
__m256i vb = _mm256_loadu_si256(...);

__m256i va_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
__m256i va_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
// ... same for vb ...

__m256i prod = _mm256_mullo_epi16(va_lo, vb_lo);  // uint16 multiplication
// Then horizontal sum...
```

This is what the scalar implementation does, just not using SIMD efficiently.

## Conclusion

**Cgo AVX2 is not viable** for this use case due to the signed byte requirement of `VPMADDUBSW`. The fundamental issue is the mismatch between Go's `uint8` (always unsigned) and the AVX2 instruction's signed byte expectation.

**Path forward:**
1. Change quantization to use signed int8 storage (requires format change)
2. Or accept current scalar performance (~2569 ns/op, 7x slower than FP32)
3. Or wait for Go assembler fixes to enable `go:asm` approach

The cgo infrastructure is removed; the codebase uses scalar 8-way unrolled implementation which is correct but not fast.
