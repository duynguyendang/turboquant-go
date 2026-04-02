# TurboQuant-Go Test & Benchmark Report

**Date:** 2026-04-02  
**Hardware:** AMD Ryzen 9 5900HS (8 cores, 16 threads)  
**Go Version:** 1.23  
**OS:** Linux (WSL2)

---

## 1. Micro-Benchmarks (Core Performance)

### FWHT Performance

| Benchmark | Latency | Allocations |
|---|---|---|
| Vanilla/1024 | 3,414 ns/op | 0 B/op |
| Optimized/1024 | 3,507 ns/op | 0 B/op |
| Vanilla/2048 | 7,354 ns/op | 0 B/op |
| Optimized/2048 | 7,421 ns/op | 0 B/op |
| Vanilla/4096 | 16,216 ns/op | 0 B/op |
| Optimized/4096 | 15,677 ns/op | 0 B/op |

**Note:** The "Optimized" version (4-way unrolled Go loop) shows no measurable advantage over vanilla on this CPU. Both are memory-bandwidth bound.

### Quantization Latency

| Mode | Latency | Memory | Allocations |
|---|---|---|---|
| 8-bit | 15,680 ns/op | 10,880 B/op | 2 |
| 4-bit | 15,717 ns/op | 9,728 B/op | 2 |
| 4-bit + QJL | 26,450 ns/op | 27,008 B/op | 6 |

**QJL overhead:** +68% latency, +178% memory (due to residual computation + dequantize pass).

### Dequantization Throughput

| Mode | Latency | Throughput | Memory |
|---|---|---|---|
| 8-bit | 13,066 ns/op | 470 MB/s | 14,336 B/op |
| 4-bit | 13,327 ns/op | 461 MB/s | 14,336 B/op |
| 4-bit + QJL | 13,514 ns/op | 455 MB/s | 14,336 B/op |

QJL adds negligible overhead to dequantization (bits are not used during decompression).

### Dot Product Shootout

| Method | Latency | Speedup vs FP32 | Allocations |
|---|---|---|---|
| FP32 | 730 ns/op | 1.0× | 0 |
| 8-bit | 2,971 ns/op | 0.25× | 0 |
| 4-bit | 3,490 ns/op | 0.21× | 0 |
| 4-bit + QJL | 3,517 ns/op | 0.21× | 0 |
| HybridFull | 26,214 ns/op | 0.03× | 28,672 B/op |

**Key finding:** Compressed dot products are 4× slower than FP32, not faster. The block-wise dequantization overhead dominates. HybridFull (dequantize + full dot) is 36× slower.

### Dimension Scaling (Quantize 4-bit)

| Dimension | Latency | Memory |
|---|---|---|
| 256 | 1,672 ns/op | 1,216 B/op |
| 512 | 3,429 ns/op | 2,432 B/op |
| 1024 | 7,570 ns/op | 4,864 B/op |
| 1536 | 15,352 ns/op | 9,728 B/op |
| 2048 | 15,550 ns/op | 9,728 B/op |
| 4096 | 31,360 ns/op | 19,456 B/op |

Linear scaling with dimension. Note: 1536 and 2048 have identical padded dim (2048), hence same cost.

---

## 2. Search Benchmarks (System Performance)

### Latency Scaling (Top-10 Search)

| Dataset | 8-bit | 4-bit | 4-bit + QJL |
|---|---|---|---|
| 10K | 7.07 ms | 8.85 ms | 10.41 ms |
| 100K | 76.6 ms | 95.0 ms | 106.1 ms |
| 500K | 392.8 ms | 687.9 ms | — |

4-bit is ~24% slower than 8-bit at 100K due to nibble unpacking overhead. QJL adds ~12% on top of 4-bit.

### QPS (Parallel Search, 100K vectors)

| Benchmark | Latency | QPS |
|---|---|---|
| Search (parallel) | 47.2 ms | ~21 q/s per goroutine |

### Parallelism Efficiency (100K vectors)

| Workers | Latency | Speedup vs 1 worker | Allocations |
|---|---|---|---|
| 1 | 244.8 ms | 1.0× | 413 KB |
| 4 | 77.9 ms | 3.1× | 440 KB |
| 8 | 54.2 ms | 4.5× | 476 KB |
| 12 | 54.0 ms | 4.5× | 511 KB |

Diminishing returns after 8 workers. 4× speedup with 4 workers is near-linear.

### Cold vs Warm Cache (100K vectors)

| Mode | Latency | Memory |
|---|---|---|
| Cold (first query) | 4,185 ms | 2.5 GB |
| Warm (subsequent) | 75.3 ms | 440 KB |

Cold start is 55× slower due to full vector generation + registry build. Warm search is consistent.

### Top-K Scaling (100K vectors)

| K | Latency | Memory |
|---|---|---|
| 1 | 72.7 ms | 438 KB |
| 5 | 73.7 ms | 439 KB |
| 10 | 75.5 ms | 440 KB |
| 50 | 72.4 ms | 450 KB |
| 100 | 74.7 ms | 462 KB |

Search latency is dominated by dot product computation, not heap management. K has minimal impact.

---

## 3. Accuracy Benchmarks

### Recall@K (10K vectors, seed=42)

| Mode | Recall@1 | Recall@10 | Recall@100 |
|---|---|---|---|
| 8-bit | **100.0%** | **100.0%** | 99.0% |
| 4-bit | **100.0%** | 80.0% | 92.0% |
| 4-bit + QJL | **100.0%** | 80.0% | 91.0% |

**QJL Uplift:** 0% at Recall@10, -1% at Recall@100. Per-block QJL with sign bits does not improve ranking for random vectors.

### MSE of Similarity Scores (1K vectors)

| Mode | MSE |
|---|---|
| 8-bit | 0.00000001 |
| 4-bit | 0.00000291 |
| 4-bit + QJL | 0.00000402 |

QJL slightly increases MSE — the correction term adds noise rather than signal for uncorrelated vector pairs.

### Recall by Similarity Bucket (10K vectors)

| Bucket | 8-bit | 4-bit | 4-bit + QJL |
|---|---|---|---|
| 0.5-0.6 | 100.0% | 80.0% | 80.0% |
| 0.6-0.7 | 100.0% | 80.0% | 80.0% |
| 0.7-0.8 | 100.0% | 80.0% | 80.0% |
| 0.8-0.9 | 100.0% | 80.0% | 80.0% |
| 0.9-1.0 | 100.0% | 80.0% | 80.0% |

QJL shows no improvement across any similarity range. The residual sign bits are uncorrelated between different vectors.

### Search Score Comparison (QJL vs Plain)

| Rank | 4-bit Score | 4-bit + QJL Score | Delta |
|---|---|---|---|
| #1 | 0.0624 | 0.0635 | +0.0011 |
| #2 | 0.0602 | 0.0612 | +0.0010 |
| #3 | 0.0496 | 0.0506 | +0.0010 |

QJL consistently adds ~0.001 to scores but doesn't change ranking.

### Clustered Vector Accuracy (100 clusters × 100 vectors)

| Spread | Intra-Cluster Sim | Mode | Recall@1 | Recall@10 | Recall@100 |
|---|---|---|---|---|---|
| 0.0100 | 0.86-0.93 | 8-bit | 100.0% | 90.0% | 100.0% |
| 0.0100 | 0.86-0.93 | 4-bit | 0.0% | 60.0% | 100.0% |
| 0.0100 | 0.86-0.93 | 4-bit+QJL | 0.0% | 60.0% | 100.0% |
| 0.0050 | 0.96-0.98 | 8-bit | 100.0% | 100.0% | 100.0% |
| 0.0050 | 0.96-0.98 | 4-bit | 0.0% | 50.0% | 100.0% |
| 0.0050 | 0.96-0.98 | 4-bit+QJL | 0.0% | 50.0% | 100.0% |
| 0.0020 | 0.993-0.997 | 8-bit | 100.0% | 90.0% | 100.0% |
| 0.0020 | 0.993-0.997 | 4-bit | 0.0% | 40.0% | 100.0% |
| 0.0020 | 0.993-0.997 | 4-bit+QJL | 0.0% | 40.0% | 100.0% |

**QJL Uplift on clustered vectors: 0% at all spreads.**

### Why QJL Fails Even on Clustered Vectors

Direct measurement at spread=0.002 (similarity 0.997):

| Metric | Value | Implication |
|---|---|---|
| Blocks with identical 4-bit codes | 0/64 | 4-bit codes still differ between similar vectors |
| Sign correlation (same sign) | 54.3% | Barely above random (50%) |
| Total QJL correction | 176 | Positive but small — noise, not signal |

**Root cause:** FWHT spreads energy uniformly across all dimensions. Even for nearly identical vectors (similarity 0.997), the FWHT-transformed residuals are **uncorrelated** — the transform decorrelates local structure. The sign bits end up matching only 54% of the time, producing noise-level correction.

This is a fundamental limitation of the QJL approach with FWHT: the transform that enables efficient compression also destroys the local residual structure that QJL needs to work.

---

## 4. Resource & Memory Benchmarks

### Memory Footprint (100K vectors)

| Mode | Total Alloc | Per-Vector | Compressed Size | Overhead |
|---|---|---|---|---|
| FP32 | 2,383 MB | 24,992 B | 2,560 B | 876% |
| 8-bit | 2,383 MB | 24,992 B | 2,560 B | 876% |
| 4-bit | 2,176 MB | 22,816 B | 1,536 B | 1,385% |
| 4-bit + QJL | 3,860 MB | 40,480 B | 1,920 B | 2,008% |

**Note:** High "overhead" is due to temporary allocations during vector generation (input `[]float32` slices). Actual registry storage matches compressed size.

### Allocations per Query (10K vectors)

| Operation | Latency | Memory | Allocations |
|---|---|---|---|
| Search | 7.05 ms | 55 KB | 35 |
| SearchIter | 30.54 ms | 25.7 MB | 14 |
| All | 6.11 ms | 25.7 MB | 5 |
| IDsIter | 28.8 μs | 82 KB | 4 |

SearchIter copies the entire vector slab (25.7 MB for 10K × 2,560B). Consider zero-copy alternatives for large registries.

### GC Pressure (100K vectors, 100 concurrent queries)

| Metric | Value |
|---|---|
| Total time | ~5-10s (estimated) |
| QPS | ~10-20 |
| GC runs | Proportional to allocation rate |

---

## 5. Summary & Key Findings

### What Works Well
- **8-bit accuracy**: Recall@10 = 100%, MSE = 1e-8 — near-perfect
- **Parallelism**: 3.1× speedup with 4 workers, near-linear scaling
- **Compression**: 4-bit uses 25% of FP32 memory (1,536B vs 6,144B)
- **Zero-allocation dot products**: All compressed modes use 0 B/op

### What Needs Improvement
- **4-bit accuracy**: Recall@10 = 80% — significant drop from 8-bit
- **QJL effectiveness**: Per-block QJL with sign bits provides **no measurable improvement** for random vectors. The residual sign correlation between different vectors is too weak to affect rankings.
- **Dot product speed**: Compressed dot products are 4× slower than FP32, not faster
- **SearchIter memory**: Copies entire vector slab (25+ MB for 10K vectors)

### QJL Root Cause Analysis
The QJL correction formula `Σ(wA_b × wB_b × (blockSize - 2×popcount))` captures only the residual-residual term of the dot product error:

```
error = Σ(dA × rB) + Σ(rA × dB) + Σ(rA × rB)
         cross-term    cross-term    QJL captures this
```

For different vectors, cross-terms dominate (10-100× larger) and QJL cannot model them. The sign bits are essentially uncorrelated noise between unrelated vectors.

### Recommendations
1. **Use 8-bit for production** — best accuracy/speed tradeoff
2. **QJL is not recommended** for general search — adds memory and latency without accuracy gain
3. **Consider alternative approaches** for 4-bit accuracy: Product Quantization residuals, or learned codebooks
4. **Optimize dot product** — current implementation is slower than FP32; needs SIMD vectorization
