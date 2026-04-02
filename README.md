# turboquant-go

**Inspired by** the TurboQuant paper — a standalone Go implementation of hybrid vector compression and search using FWHT (Fast Walsh-Hadamard Transform) preconditioning with block-wise 4/8-bit quantization.

> This project takes inspiration from the TurboQuant research but implements its own approach. It is not a direct reproduction of the paper's methods.

## Features

- **Lossy compression**: 1536-dim float32 vectors compress to 2,560 bytes (8-bit) or 1,536 bytes (4-bit) vs 6,144 bytes raw
- **Compressed dot product**: Compute cosine similarity directly on compressed vectors without full decompression
- **Parallel search**: Multi-worker top-K search with configurable parallelism (near-linear scaling up to 8 workers)
- **AVX2 SIMD**: Shuffle-based 4-bit dequantization using `_mm256_shuffle_epi8` for fast nibble decoding
- **Go 1.23 iterators**: `All()`, `AllDecompressed()`, `IDsIter()`, `SearchIter()` for zero-allocation iteration
- **Zero dependencies**: No external dependencies (pure Go)

## Installation

```bash
go get github.com/duynguyendang/turboquant-go
```

Requires Go 1.23+.

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/duynguyendang/turboquant-go"
)

func main() {
    // Create registry with default config (1536-dim, 8-bit)
    reg, _ := turboquant.NewRegistry(nil)

    // Add vectors (IDs must be unique, vectors must match FullDim)
    reg.Add(1, generateVector(1536))
    reg.Add(2, generateVector(1536))

    // Search top-5 most similar
    results, _ := reg.Search(queryVector, 5)
    for _, r := range results {
        fmt.Printf("ID: %d, Score: %.4f\n", r.ID, r.Score)
    }
}
```

## Configuration

```go
cfg := &turboquant.Config{
    FullDim:         1536,          // Original vector dimension
    HybridBitWidth:  8,             // 4 or 8 bit quantization
    HybridBlockSize: 32,            // Block size (must be divisible by 8)
    NumWorkers:      4,             // Parallel workers for search
    VectorCapacity:  100000,        // Pre-allocated vector slots
    EnableQJL:       false,         // Enable 1-bit residual correction (4-bit only)
}

reg, _ := turboquant.NewRegistry(cfg)
```

## Core API

### Compression

```go
// Compress a vector
compressed := turboquant.QuantizeHybrid(vec, nil) // nil uses default config

// Decompress a vector
decompressed := turboquant.DequantizeHybrid(compressed, 1536, nil)

// Compute dot product on compressed vectors (no decompression needed)
score := turboquant.DotProductHybrid(vecA_compressed, vecB_compressed, 1536, nil)
```

### Registry

```go
reg, _ := turboquant.NewRegistry(nil)

reg.Add(id uint64, vec []float32) error          // Add a vector
data, ok := reg.Get(id uint64) ([]byte, bool)    // Get compressed vector
vec, ok := reg.GetDecompressed(id uint64) ([]float32, bool)  // Get decompressed
reg.Delete(id uint64) bool                        // Delete a vector
n := reg.Count()                                  // Count vectors
exists := reg.Has(id uint64) bool                 // Check existence
ids := reg.IDs()                                  // Get all IDs
```

### Search

```go
// Search returns top-K most similar (by cosine similarity)
results, _ := reg.Search(queryVector, 10)
```

### Iterators (Go 1.23+)

```go
// Iterate over all vectors (compressed)
for id, vec := range reg.All() { ... }

// Iterate over all vectors (decompressed)
for id, vec := range reg.AllDecompressed() { ... }

// Iterate over IDs only
for id := range reg.IDsIter() { ... }

// Stream search results (supports early exit)
for result := range reg.SearchIter(query, 10) {
    if result.Score < 0.5 { break }
}
```

## How It Works

1. **Pad** to next power of 2 (1536 → 2048)
2. **FWHT** — Fast Walsh-Hadamard Transform spreads energy uniformly across dimensions
3. **Normalize** — scale by 1/√N to maintain unitary property
4. **Block-wise quantize** — divide into 64 blocks of 32 elements, quantize each independently with per-block scale and zero-point

```
Input: 1536-d float32 (6,144 bytes)
  → Pad to 2048
  → FWHT (in-place, O(N log N))
  → Normalize
  → Block-wise quantize (64 blocks × 32 elements)
  → Output: 2,560 bytes (8-bit) or 1,536 bytes (4-bit)
```

### Block Layout

**8-bit:**
```
Per block: [scale:4B][zero:4B][q_0:1B]...[q_31:1B]
Total: 40 bytes/block × 64 blocks = 2,560 bytes
```

**4-bit:**
```
Per block: [scale:4B][zero:4B][q_0‖q_1:1B]...[q_30‖q_31:16B]
Total: 24 bytes/block × 64 blocks = 1,536 bytes
```

### Compressed Dot Product

Dot product is computed without decompression:

```
dot(a, b) = Σ_blocks (scale_a·scale_b·Σ(q_a·q_b)
                     + scale_a·zero_b·Σ(q_a)
                     + scale_b·zero_a·Σ(q_b)
                     + blockSize·zero_a·zero_b)
```

## Optimizations

### AVX2 Shuffle Dequantization (4-bit)
- `_mm256_shuffle_epi8` unpacks 32 nibbles simultaneously
- `VPMOVZXBD` + `VCVTDQ2PS` for byte→float32 conversion in SIMD

### Loop Unrolling
- **FWHT**: 4-way butterfly unrolling
- **Block dot products**: 8-way accumulator unrolling
- **Quantize/Unpack**: 4-way and 8-way unrolling

### Architecture Dispatch
- `fwht_amd64.go` — AMD64-optimized FWHT (build tags)
- `dotprod_amd64.go` — AMD64-optimized dot products with multi-accumulator unrolling
- `dequant_amd64.s` — AVX2 assembly for 4-bit/8-bit dequantization
- `*_generic.go` — portable fallbacks for non-AMD64

Disable SIMD: `go build -tags noasm`

## Benchmark Results

**Hardware:** AMD Ryzen 9 5900HS (8 cores, 16 threads)  
**Go:** 1.23 | **OS:** Linux

### Micro-Benchmarks (1536-dim)

| Operation | Latency | Memory | Allocations |
|---|---|---|---|
| Quantize 8-bit | 15.7 μs | 10.9 KB | 2 |
| Quantize 4-bit | 15.7 μs | 9.7 KB | 2 |
| Dequantize 8-bit | 13.1 μs (470 MB/s) | 14.3 KB | 2 |
| Dequantize 4-bit | 13.3 μs (461 MB/s) | 14.3 KB | 2 |
| Dot product FP32 | 0.73 μs | 0 B | 0 |
| Dot product 8-bit | 2.97 μs | 0 B | 0 |
| Dot product 4-bit | 3.49 μs | 0 B | 0 |
| HybridFull (dequantize+dot) | 26.2 μs | 28.7 KB | 4 |

### Search Latency (Top-10)

| Dataset | 8-bit | 4-bit | 4-bit + QJL |
|---|---|---|---|
| 10K vectors | 7.1 ms | 8.8 ms | 10.4 ms |
| 100K vectors | 76.6 ms | 95.0 ms | 106.1 ms |
| 500K vectors | 392.8 ms | 687.9 ms | — |

### Parallelism (100K vectors, 8-bit)

| Workers | Latency | Speedup |
|---|---|---|
| 1 | 244.8 ms | 1.0× |
| 4 | 77.9 ms | 3.1× |
| 8 | 54.2 ms | 4.5× |
| 12 | 54.0 ms | 4.5× |

### FWHT Performance

| Dimension | Latency |
|---|---|
| 1024 | 3.4 μs |
| 2048 | 7.4 μs |
| 4096 | 15.7 μs |

## Compression Ratio

| Format | Size | Ratio |
|---|---|---|
| float32 (1536-d) | 6,144 B | 100% |
| 8-bit | 2,560 B | 41.7% |
| 4-bit | 1,536 B | 25.0% |
| 4-bit + QJL | 1,920 B | 31.3% |

## Accuracy

### Reconstruction (Cosine Similarity)

| Quantization | Similarity |
|---|---|
| 8-bit | 0.999+ |
| 4-bit | 0.99+ |

### Search Recall@10 (10K vectors)

| Mode | Random vectors | Clustered (spread=0.01) |
|---|---|---|
| 8-bit | 100.0% | 90.0% |
| 4-bit | 80.0% | 60.0% |
| 4-bit + QJL | 80.0% | 60.0% |

**Note:** QJL (1-bit residual correction) provides no measurable improvement for either random or clustered vectors. The FWHT transform decorrelates residuals, making sign bits uncorrelated between different vectors. See `docs/test-report.md` for detailed analysis.

## Running Benchmarks

```bash
# All benchmarks
go test -bench=. -benchmem

# Search only
go test -bench=Search -benchmem

# Accuracy tests
go test -v -run "TestRecall|TestMSE"

# With race detector
go test -race -count=1

# Short mode (skips heavy tests)
go test -short
```

## Project Structure

```
├── config.go           # Configuration types
├── registry.go         # Vector storage and search
├── search.go           # Search algorithms
├── hybrid.go           # FWHT + block-wise quantization
├── dotprod.go          # Dot product dispatch
├── dotprod_amd64.go    # AMD64 dot product (unrolled)
├── dotprod_generic.go  # Generic fallback
├── dotprod_batch_amd64.go     # Batch dot product (AMD64)
├── dotprod_batch_generic.go   # Batch dot product (generic)
├── dequant_amd64.go    # Dequantization dispatch (AMD64)
├── dequant_amd64.s     # AVX2 assembly for dequantization
├── dequant_generic.go  # Generic dequantization
├── fwht.go             # FWHT implementation
├── fwht_amd64.go       # AMD64 FWHT (unrolled)
├── lut.go              # Lookup table optimizations
├── math.go             # Math utilities (L2Normalize, DotProduct)
├── qjl.go              # QJL 1-bit residual correction
├── qjl_amd64.go        # QJL dispatch (AMD64)
├── qjl_generic.go      # QJL dispatch (generic)
├── pool.go             # Memory pool utilities
├── utils.go            # Byte encoding helpers
├── errors.go           # Error types
├── turboquant_test.go  # Tests and basic benchmarks
├── benchmark_test.go   # Comprehensive benchmarks
└── docs/
    ├── test-report.md           # Full benchmark report
    └── qjl-implementation.md    # QJL design documentation
```

## License

MIT
