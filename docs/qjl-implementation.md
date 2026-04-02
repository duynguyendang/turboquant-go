# QJL (Quantized Johnson-Lindenstrauss) 1-Bit Residual Correction

## Overview

QJL is an optional enhancement for 4-bit quantization that stores 1-bit sign information about the quantization residual. This allows the compressed dot product to partially recover information lost during quantization.

## How It Works

### Encoding (Quantize)

1. Compress the vector using standard 4-bit block-wise quantization (FWHT + block quantize)
2. Dequantize back to float32 to get the approximate vector
3. Compute the residual: `residual[i] = original[i] - dequantized[i]`
4. Store the **sign** of each residual as a single bit (1 if ≥ 0, 0 if < 0)
5. Store a scalar **weight** = max(mean(|residual|), RMS(residual))

### Decoding (Dot Product)

The compressed dot product is corrected using:

```
Score = Score_4bit + (w_q × w_v) × (dim - 2 × popcount(bits_q ⊕ bits_v)) / dim
```

Where:
- `w_q`, `w_v` = stored weights for query and database vectors
- `bits_q ⊕ bits_v` = XOR of sign bit arrays
- `popcount` = number of set bits (Hamming distance)
- `dim` = padded dimension (2048 for 1536-dim input)

The term `(dim - 2 × popcount)` equals the number of matching signs minus mismatching signs, ranging from `-dim` to `+dim`.

## Storage Layout

For 1536-dim vectors (padded to 2048):

```
[4-bit blocks: 1,536B] [weight: 4B] [QJL bits: 192B]
                       └────────────┬─────────────────┘
                         196 bytes overhead

Total: 1,732 bytes (vs 1,536 bytes plain 4-bit, 6,144 bytes FP32)
```

- **Weight**: `float32` at offset 1,536
- **QJL bits**: 24 × `uint64` (1,536 bits packed) at offset 1,540

## Performance

| Metric | 4-bit Plain | 4-bit + QJL | Change |
|--------|------------|-------------|--------|
| Dot product latency | ~3.1 μs | ~3.4 μs | +10% |
| Search 100K vectors | ~101 ms | ~94 ms | -7% |
| Memory per vector | 1,536 B | 1,732 B | +12.8% |
| Compression vs FP32 | 75.0% | 71.8% | - |

QJL adds negligible overhead to search because:
- Popcount uses `math/bits.OnesCount64` which compiles to hardware `POPCNT` on amd64
- Only 24 XOR + popcount operations per vector comparison
- Stack-allocated buffers (no heap allocation)

## Usage

```go
cfg := &turboquant.Config{
    FullDim:         1536,
    HybridBitWidth:  4,
    HybridBlockSize: 32,
    EnableQJL:       true,  // enable QJL
}
reg, _ := turboquant.NewRegistry(cfg)
```

Or for standalone operations:

```go
cfg := &turboquant.HybridConfig{
    BitWidth:  4,
    BlockSize: 32,
    EnableQJL: true,
}
compressed := turboquant.QuantizeHybrid(vec, cfg)
score := turboquant.DotProductHybrid(compA, compB, 1536, cfg)
```

## Design Notes

### Why Original-Space Residuals

Residuals are computed after inverse FWHT (in the original vector space) rather than in FWHT space. This is because:

1. FWHT spreads energy uniformly, making residuals small and uncorrelated
2. Original-space residuals capture structured quantization error
3. The sign bits are more meaningful when applied to the final vector

### Why Not Apply QJL to Dequantization

QJL correction is applied only during compressed dot product, not during `DequantizeHybrid`. Adding ±weight to each dimension before inverse FWHT would spread the correction everywhere, degrading reconstruction quality.

### Single Global Weight

A single `float32` weight is used for all dimensions rather than per-block weights. This keeps overhead minimal (4 bytes vs 256 bytes for per-block). The trade-off is that dimensions with large residuals dominate the weight, while small-residual dimensions get over-corrected.

### Limitations

QJL provides the most benefit when:
- Comparing very similar vectors (high cosine similarity)
- The quantization error has structured patterns

For random/unrelated vectors, the residual signs are uncorrelated and the correction term averages to near zero.

## Architecture Support

| Platform | Popcount Implementation |
|----------|------------------------|
| amd64 | `math/bits.OnesCount64` → hardware `POPCNT` |
| arm64 | `math/bits.OnesCount64` → software fallback |
| Other | `math/bits.OnesCount64` → software fallback |

Future work: AVX-512 `VPOPCNTQ` for processing 8 uint64s simultaneously.
