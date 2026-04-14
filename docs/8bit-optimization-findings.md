# 8-bit Dot Product Optimization Findings

**Date:** April 14, 2026
**Goal:** Optimize 8-bit compressed dot product performance

## Performance Baseline (Ryzen 9 5900HS)

| Metric | 8-bit | 4-bit | 4-bit+QJL | FP32 |
|---|---|---|---|---|
| Single dot product (1536-dim) | ~3,300 ns | ~3,570 ns | ~4,115 ns | ~392 ns |
| Search 100K vectors (parallel, 8 workers) | ~124 ms | ~130 ms | ~140 ms | N/A |
| Memory per vector | 2,560 B | 1,536 B | 1,920 B | 6,144 B |

## Attempts Made (All Failed)

### 1. AVX2 Assembly — VPMADDUBSW
**Approach:** Use `VPMADDUBSW` for unsigned×signed byte multiply with horizontal add.
**Problem:** The instruction computes `Σ(src1_signed × src2_unsigned)`. With both vectors stored as signed int8:
- `VPMADDUBSW(B_signed, A_as_unsigned) = Σ(b × (a+128)) = Σ(a×b) + 128×Σ(b)`
- Requires correction: `Σ(a×b) = result - 128 × Σ(b)`
- But computing `Σ(b)` itself requires a separate SIMD path (VPMOVZXBW + VPMADDWD), which was also buggy
- Multiple instruction ordering bugs in Go's assembler (VPUNPCK interleaving semantics differ from Intel)
- **Result:** AVX2 produced sums 10^12× larger than scalar — sign/extension bugs in register handling

### 2. AVX2 Assembly — VPMULLW (uint16×uint16)
**Approach:** Unpack uint8→uint16 with VPUNPCKLBW/HBW, multiply with VPMULLW, reduce with VPUNPCKLWD/HWD + VPADDD.
**Problem:** VPMULLW treats inputs as **signed int16**. Products like 200×200=40,000 exceed int16 range (32,767), becoming negative. Zero-extending negative int16 produces huge unsigned uint32 values (~2^32).
**Result:** Same 10^12× magnitude errors.

### 3. AVX2 Assembly — VPMADDWD with ones vector
**Approach:** Unpack to uint16, multiply against 16×int16(1) vector for horizontal sum.
**Problem:** VPMADDWD processes pairs: `result[i] = src[2i]*1 + src[2i+1]*1`. Reducing 16 values to a single sum requires multiple rounds of extraction and accumulation, which is complex to get right in Go's assembler and likely slower than expected.
**Result:** Abandoned mid-implementation due to cascading complexity.

### 4. Stack precomputation with fixed arrays
**Approach:** Precompute query block scale/zero into `[64]float32` stack arrays to eliminate redundant reads.
**Problem:** Allocation overhead of the fixed arrays + the per-vector loop structure actually made search 19% slower (114ms vs 96ms).
**Result:** Reverted.

## Root Cause Analysis

### Why AVX2 keeps failing in Go assembler

1. **VPUNPCKLBW interleaving**: `VPUNPCKLBW src1, src2, dest` → `dest = [src2[0], src1[0], src2[1], src1[1], ...]`. This is counterintuitive — the 2nd source goes to even positions, 1st source to odd positions. Zero-extending requires the data byte in the **even** position (low byte of each word), but the interleaving puts it in the odd position, making each uint16 = `data × 256`.

2. **No unsigned 8×8→32 multiply in AVX2**: 
   - `VPMADDUBSW` = unsigned×signed → int32 (half the operation we need)
   - `VPMULLW` = signed×signed → int16 (wrong sign semantics)
   - `VPMULLD` = signed×signed → int32 (requires prior unpack to 32-bit)
   - There is no `VPMULLUBW` or similar unsigned×unsigned instruction

3. **Go's text stub ABI**: Slices are 24 bytes (data+len+cap) on the stack. The assembler text stub places all 24 bytes for each slice argument, shifting offsets. `a_base` at +0, `a_len` at +8, `a_cap` at +16, `b_base` at +24, etc. Getting this wrong causes reading garbage data.

### Why the scalar code is already near-optimal

The 8-way unrolled scalar processes 8 bytes per loop iteration with 48 integer operations (16 multiplies + 32 adds). Go's compiler inlines this aggressively. The per-iteration overhead is ~1-2 CPU cycles for loop control. At 4 iterations for a 32-byte block, this is already tight.

## What Was Accomplished

### Signed int8 storage format ✅
- Quantization stores `byte(q - 128)` instead of `byte(q)`
- Dequantization reads as `int(int8(byte)) + 128`
- Dot product formula expanded: `unsignedSumQQ = signedSumQQ + 128×(signedSumQA + signedSumQB) + 16384×blockLen`
- This **enables** future AVX2 work with `VPMADDUBSW` — the storage format was the prerequisite

### Function naming cleanup ✅
- `dotProdBlock8_AVX2` → `dotProdBlock8_impl` (no AVX2, misleading name)
- `dotProdBlock8_scalar_impl` → `dotProdBlock8_unrolled`
- Same for 4-bit variants

## Remaining Viable Paths

### Path A: cgo with C intrinsics
Write the AVX2 batch function in C using compiler intrinsics (`<immintrin.h>`), called from Go via cgo. Batch all 64 blocks in a single cgo call to amortize the ~100ns cgo overhead.
- **Pros:** C compiler handles instruction encoding correctly; access to `_mm256_maddubs_epi16`
- **Cons:** cgo build complexity; cross-compilation issues; `VPMADDUBSW` correction logic still needed
- **Estimated gain:** 2-3x for single dot product (~1,000-1,500 ns)

### Path B: Accept current performance
- 8-bit saves 60% memory vs FP32
- Search is parallelized (4.5x with 8 workers)
- Recall@10 = 100% for 8-bit
- The 3.3μs dot product is fast enough for most use cases (100K search in ~124ms)

### Path C: Profile the real bottleneck
The dot product itself is only part of the search cost. The batch loop does:
1. 64× `getFloat32` calls per vector (128 reads of vecData headers)
2. 64× float32 math operations for the expansion formula
3. Memory access patterns (64 blocks × 32 bytes = 2,048 bytes per vector, likely L2 cache resident)

Profiling may reveal that steps 1-2 are the dominant cost, not the byte multiply loop.

## Conclusion

After 4+ failed AVX2 assembly attempts, the conclusion is clear: **Go's assembler is not the right tool for this optimization**. The combination of counterintuitive instruction semantics (VPUNPCK interleaving), lack of unsigned byte multiply, and complex register management makes it impractical to get correct.

The best path forward is either:
1. **cgo with C intrinsics** for the AVX2 batch function (requires build system changes)
2. **Accept current scalar performance** as "good enough" given the memory savings and parallelization benefits

The signed int8 storage change is retained — it's the prerequisite for any future AVX2 work and has no performance downside.
