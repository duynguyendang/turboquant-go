# turboquant-go

A standalone Go implementation of **TurboQuant** - a hybrid vector compression and search library using FWHT (Fast Walsh-Hadamard Transform) preconditioning with block-wise 4/8-bit quantization.

## Features

- **Lossy compression**: 1536-dim float32 vectors compress to 2,560 bytes (8-bit) or 1,536 bytes (4-bit) vs 6,144 bytes raw
- **Fast dot product**: Compute cosine similarity directly on compressed vectors without full decompression
- **Parallel search**: Multi-worker top-K search with configurable parallelism
- **Zero dependencies**: No external dependencies (pure Go)

## Installation

```go
go get github.com/duynguyendang/turboquant-go
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/duynguyendang/turboquant-go"
)

func main() {
    // Create registry with default config (1536-dim, 8-bit)
    reg, err := turboquant.NewRegistry(nil)
    if err != nil {
        panic(err)
    }

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

// Compute dot product on compressed vectors (fast!)
score := turboquant.DotProductHybrid(vecA_compressed, vecB_compressed, 1536, nil)
```

### Registry

```go
reg, _ := turboquant.NewRegistry(nil)

// Add vectors
reg.Add(id uint64, vec []float32) error

// Get compressed vector
data, ok := reg.Get(id uint64) ([]byte, bool)

// Get decompressed vector
vec, ok := reg.GetDecompressed(id uint64) ([]float32, bool)

// Delete vector
reg.Delete(id uint64) bool

// Count vectors
n := reg.Count()

// Check existence
exists := reg.Has(id uint64) bool

// Get all IDs
ids := reg.IDs()
```

### Search

```go
// Search returns top-K most similar (by cosine similarity)
results, _ := reg.Search(queryVector, 10)

// Results contain ID and Score
for _, r := range results {
    fmt.Printf("ID: %d, Score: %.6f\n", r.ID, r.Score)
}
```

## Compression Details

TurboQuant uses a hybrid approach:

1. **Pad to power of 2**: Input vectors are padded to the next power of 2 dimension
2. **FWHT**: Apply Fast Walsh-Hadamard Transform to spread energy uniformly
3. **Normalize**: Scale by 1/√N to maintain unitary property
4. **Block-wise quantization**: Divide into blocks, quantize each block independently

```
Input: 1536-d float32
  → Pad to 2048
  → FWHT (in-place, O(N log N))
  → Normalize
  → Block-wise quantize (64 blocks × 32 elements)
  → Output: 2,560 bytes (8-bit) or 1,536 bytes (4-bit)
```

### Block Layout (8-bit)

```
Per block: [scale:4B][zero:4B][q_0:1B][q_1:1B]...[q_31:1B]
Total: 8 + 32 = 40 bytes per block
64 blocks × 40 bytes = 2,560 bytes
```

### Compressed Dot Product

The dot product is computed without full decompression using the formula:

```
dot(a, b) = Σ_blocks (scale_a * scale_b * Σ(q_a_i * q_b_i)
                      + scale_a * zero_b * Σ(q_a_i)
                      + scale_b * zero_a * Σ(q_b_i)
                      + block_size * zero_a * zero_b)
```

## Optimizations

### SIMD-Inspired Loop Unrolling
- **FWHT**: 4-way butterfly unrolling for better CPU pipelining
- **Block dot products**: 8-way accumulator unrolling for reduced loop overhead
- **Quantize/Unpack**: 4-way and 8-way unrolling for parallel processing

### Look-Up Tables (LUT)
- **Dequantization LUT**: Pre-computed byte→float32 mappings (256 entries for 8-bit, 16 for 4-bit)
- **Quantization LUT**: Fast rounding with clamping using bit operations
- **SIMD-style indexing**: Avoid per-element clamp function calls

### Architecture Dispatch
- `fwht_amd64.go`: AMD64-optimized FWHT (auto-selected via build tags)
- `dotprod_amd64.go`: AMD64-optimized dot products
- `dotprod_generic.go`: Portable fallback for non-AMD64 platforms

To disable SIMD optimizations: `go build -tags noasm`

## Benchmark Results

Hardware: AMD Ryzen 9 5900HS (8 cores)

```
Benchmark Quantization (1536-dim):
BenchmarkQuantize-8        13,852 ns/op   10,880 B/op   2 allocs
BenchmarkQuantize4Bit-8    15,709 ns/op    9,728 B/op   2 allocs

Benchmark Dot Product (1536-dim):
BenchmarkDotProductFull-8        337 ns/op     0 B/op   0 allocs
BenchmarkDotProductHybrid-8   2,332 ns/op     0 B/op   0 allocs
BenchmarkDotProductHybrid4Bit 2,282 ns/op     0 B/op   0 allocs

Benchmark Search (top-10):
BenchmarkSearch1K-8        747,987 ns/op   13,913 B/op   28 allocs
BenchmarkSearch10K-8     6,633,130 ns/op   13,861 B/op   28 allocs

Benchmark Dequantization:
BenchmarkDequantize-8        11,035 ns/op   14,336 B/op   2 allocs
BenchmarkDequantize4Bit-8    10,705 ns/op   14,336 B/op   2 allocs

Benchmark FWHT (SIMD-optimized):
BenchmarkFWHT-8          6,600 ns/op   0 B/op   0 allocs (2048-dim)
BenchmarkFWHT1024-8     3,025 ns/op   0 B/op   0 allocs
BenchmarkFWHT2048-8     6,494 ns/op   0 B/op   0 allocs
BenchmarkFWHT4096-8    13,941 ns/op   0 B/op   0 allocs
```

### Performance Summary

| Operation | Latency | Memory |
|-----------|---------|--------|
| Quantize (1536-d) | ~13.9 μs | 10.9 KB/op |
| Dequantize (1536-d) | ~11 μs | 14.3 KB/op |
| Compressed dot product | ~2.3 μs | 0 B/op |
| Search 10K vectors | ~6.6 ms | 13.9 KB/op |
| FWHT (2048-dim) | ~6.5 μs | 0 B/op |

### Compression Ratio

| Format | Original | Compressed | Ratio |
|--------|----------|------------|-------|
| float32 (1536-d) | 6,144 B | - | 100% |
| TurboQuant 8-bit | 6,144 B | 2,560 B | 41.7% |
| TurboQuant 4-bit | 6,144 B | 1,536 B | 25.0% |

## Accuracy

| Quantization | Cosine Similarity |
|--------------|-------------------|
| 8-bit | 0.999+ |
| 4-bit | 0.99+ |

## Running Benchmarks

```bash
# Run all benchmarks
go test -bench=. -benchmem

# Run specific benchmark
go test -bench=Search -benchmem

# Run with more iterations
go test -bench=Search -benchmem -benchtime=5s -count=10
```

## License

MIT
