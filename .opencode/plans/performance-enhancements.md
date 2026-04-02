# Performance Enhancement Plan for turboquant-go

## 1. Fix dotProdBlock4_SIMD variable shadowing bug

**File:** `dotprod_amd64.go` (lines 70-117)

**Problem:** Line 80 `qa1 := int64(byteA >> 4)` shadows the outer `qa1` variable declared at line 72. The shadowed variable's value is never propagated back, causing incorrect `sumQA` results.

**Fix:** Rename the local variables to avoid shadowing:

```go
// dotProdBlock4_SIMD computes sumQQ, sumQA, sumQB for 4-bit blocks.
func dotProdBlock4_SIMD(a, b []byte, blockLen int) (sumQQ, sumQA, sumQB int64) {
	var qq0, qq1 int64
	var qa0, qa1 int64
	var qb0, qb1 int64

	i := 0
	for ; i+2 <= blockLen; i += 2 {
		byteA := a[i/2]
		byteB := b[i/2]

		hiA := int64(byteA >> 4)
		hiB := int64(byteB >> 4)
		qq0 += hiA * hiB
		qa0 += hiA
		qb0 += hiB

		loA := int64(byteA & 0x0F)
		loB := int64(byteB & 0x0F)
		qq1 += loA * loB
		qa1 += loA
		qb1 += loB
	}

	sumQQ = qq0 + qq1
	sumQA = qa0 + qa1
	sumQB = qb0 + qb1

	// Handle remainder (unchanged)
	for ; i < blockLen; i++ {
		byteA := a[i/2]
		byteB := b[i/2]
		if i%2 == 0 {
			qa := int64(byteA >> 4)
			qb := int64(byteB >> 4)
			sumQQ += qa * qb
			sumQA += qa
			sumQB += qb
		} else {
			qa := int64(byteA & 0x0F)
			qb := int64(byteB & 0x0F)
			sumQQ += qa * qb
			sumQA += qa
			sumQB += qb
		}
	}

	return
}
```

---

## 2. Wire up AVX2 batch assembly in dotprod_batch_amd64.go

**File:** `dotprod_batch_amd64.go`

**Problem:** The AVX2 assembly functions `dotProdBlock8_AVX2_batch` and `dotProdBlock4_AVX2_batch` in `dotprod_batch_amd64.s` are never called. The batch dot product loops process one byte at a time via the scalar `dotProdBlock8`/`dotProdBlock4` functions.

**Fix:** Replace the per-byte inner loop with calls to the 32-byte AVX2 batch functions.

Add extern declarations at the top of `dotprod_batch_amd64.go`:

```go
//go:noescape
func dotProdBlock8_AVX2_batch(a, b []byte) (sumQQ, sumQA, sumQB int64)

//go:noescape
func dotProdBlock4_AVX2_batch(a, b []byte, blockLen int) (sumQQ, sumQA, sumQB int64)
```

Rewrite `dotProductHybridBatch8`:

```go
func dotProductHybridBatch8(query []byte, vectors []byte, numVectors int, vectorSize int, numBlocks int, blockSize int, cfg *HybridConfig, scores []float32) {
	paddedDim := numBlocks * blockSize

	for i := 0; i < numVectors; i++ {
		vecStart := i * vectorSize
		vecData := vectors[vecStart : vecStart+vectorSize]

		var totalSum float32
		offsetA := 0
		offsetB := 0

		for block := 0; block < numBlocks; block++ {
			blockLen := blockSize
			if (block+1)*blockSize > paddedDim {
				blockLen = paddedDim - block*blockSize
			}

			scaleA := getFloat32(query[offsetA:])
			zeroA := getFloat32(query[offsetA+4:])
			scaleB := getFloat32(vecData[offsetB:])
			zeroB := getFloat32(vecData[offsetB+4:])

			offsetA += 8
			offsetB += 8

			var sumQQ, sumQA, sumQB int64
			if blockLen >= 32 {
				sumQQ, sumQA, sumQB = dotProdBlock8_AVX2_batch(
					query[offsetA:offsetA+blockLen],
					vecData[offsetB:offsetB+blockLen],
				)
			} else {
				sumQQ, sumQA, sumQB = dotProdBlock8(
					query[offsetA:offsetA+blockLen],
					vecData[offsetB:offsetB+blockLen],
				)
			}
			offsetA += blockSize
			offsetB += blockSize

			totalSum += scaleA*scaleB*float32(sumQQ) +
				scaleA*zeroB*float32(sumQA) +
				scaleB*zeroA*float32(sumQB) +
				float32(blockLen)*zeroA*zeroB
		}

		scores[i] = totalSum
	}
}
```

Rewrite `dotProductHybridBatch4` similarly:

```go
func dotProductHybridBatch4(query []byte, vectors []byte, numVectors int, vectorSize int, numBlocks int, blockSize int, cfg *HybridConfig, scores []float32) {
	paddedDim := numBlocks * blockSize

	for i := 0; i < numVectors; i++ {
		vecStart := i * vectorSize
		vecData := vectors[vecStart : vecStart+vectorSize]

		var totalSum float32
		offsetA := 0
		offsetB := 0

		for block := 0; block < numBlocks; block++ {
			blockLen := blockSize
			if (block+1)*blockSize > paddedDim {
				blockLen = paddedDim - block*blockSize
			}

			scaleA := getFloat32(query[offsetA:])
			zeroA := getFloat32(query[offsetA+4:])
			scaleB := getFloat32(vecData[offsetB:])
			zeroB := getFloat32(vecData[offsetB+4:])

			offsetA += 8
			offsetB += 8

			var sumQQ, sumQA, sumQB int64
			if blockLen >= 32 {
				sumQQ, sumQA, sumQB = dotProdBlock4_AVX2_batch(
					query[offsetA:offsetA+blockLen/2],
					vecData[offsetB:offsetB+blockLen/2],
					blockLen,
				)
			} else {
				sumQQ, sumQA, sumQB = dotProdBlock4(
					query[offsetA:offsetA+blockLen/2],
					vecData[offsetB:offsetB+blockLen/2],
					blockLen,
				)
			}
			offsetA += blockSize / 2
			offsetB += blockSize / 2

			totalSum += scaleA*scaleB*float32(sumQQ) +
				scaleA*zeroB*float32(sumQA) +
				scaleB*zeroA*float32(sumQB) +
				float32(blockLen)*zeroA*zeroB
		}

		scores[i] = totalSum
	}
}
```

---

## 3. Wire up dequantizeBlock8_AVX2_32 assembly

**File:** `dequant_amd64.go`

**Problem:** `dequantizeBlock8_AVX2` falls back to the scalar LUT version. The assembly `dequantizeBlock8_AVX2_32` exists but is never called.

**Fix:** Add the same 32-element loop pattern as the 4-bit version:

```go
func dequantizeBlock8_AVX2(data []byte, vec []float32, scale, zero float32, start, end int) int {
	blockLen := end - start
	if blockLen < 32 {
		return dequantizeBlock8_LUT(data, vec, scale, zero, start, end)
	}

	processed := 0
	i := 0
	dataOffset := 0

	for i+32 <= blockLen {
		dequantizeBlock8_AVX2_32(
			data[dataOffset:],
			vec[start+i:],
			scale,
			zero,
		)
		i += 32
		dataOffset += 32
		processed += 32
	}

	remaining := dequantizeBlock8_LUT(data[dataOffset:], vec[start+i:], scale, zero, 0, blockLen-i)
	processed += remaining

	return processed
}
```

---

## 4. Fix SearchIter full-copy memory issue

**File:** `search.go` (lines 165-224)

**Problem:** `SearchIter` copies the entire vectors slice (`make([]byte, len(r.vectors))` + `copy`). For 100K vectors at 2,560 bytes each, that's ~256MB allocated and copied per search.

**Fix:** Use a snapshot-based approach that only copies the metadata (IDs) and holds a reference to the underlying slice without copying:

```go
// SearchIter returns an iter.Seq that yields SearchResult pairs ordered by score descending.
func (r *Registry) SearchIter(query []float32, k int) iter.Seq[SearchResult] {
	if k <= 0 {
		return func(yield func(SearchResult) bool) {}
	}

	r.mu.RLock()
	numVectors := r.totalVectors
	if numVectors == 0 {
		r.mu.RUnlock()
		return func(yield func(SearchResult) bool) {}
	}

	hybridQuery := QuantizeHybrid(query, r.hybridCfg)
	vectorSize := r.vectorSize
	revMap := make([]uint64, len(r.revMap))
	copy(revMap, r.revMap)
	// Snapshot the vectors slice header only (no data copy)
	vectors := r.vectors
	dim := r.config.FullDim
	cfg := r.hybridCfg
	r.mu.RUnlock()

	return func(yield func(SearchResult) bool) {
		h := make(scoreHeap, 0, k)
		scores := make([]float32, numVectors)

		dotProductHybridBatch(hybridQuery, vectors, numVectors, vectorSize, dim, cfg, scores)

		for idx := 0; idx < numVectors; idx++ {
			score := scores[idx]

			if len(h) < k {
				h = append(h, scoreIndex{score: score, idx: idx})
				if len(h) == k {
					heap.Init(&h)
				}
			} else if score > h[0].score {
				h[0] = scoreIndex{score: score, idx: idx}
				heap.Fix(&h, 0)
			}
		}

		sorted := make([]scoreIndex, len(h))
		copy(sorted, h)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].score > sorted[j].score
		})

		for _, si := range sorted {
			if si.idx < len(revMap) {
				if !yield(SearchResult{
					ID:    revMap[si.idx],
					Score: si.score,
				}) {
					return
				}
			}
		}
	}
}
```

**Tradeoff:** This removes the defensive copy. If vectors are added/deleted concurrently after the RLock is released, the iterator may see inconsistent data. This is acceptable for most search use cases (eventual consistency) and matches the pattern used by `All()` which also copies revMap but not vectors data.

---

## 5. Optimize SearchWithLimit goroutine/channel overhead

**File:** `search.go` (lines 40-94)

**Problem:** Each search spawns `numWorkers` goroutines with a buffered channel. For small datasets, goroutine creation + channel send/recv overhead dominates.

**Fix:** Replace with `sync.WaitGroup` + pre-allocated result slice:

```go
// SearchWithLimit finds the top-k most similar vectors to the query.
func (r *Registry) SearchWithLimit(query []float32, k int) ([]SearchResult, error) {
	if k <= 0 {
		return nil, nil
	}

	r.mu.RLock()
	numVectors := r.totalVectors
	if numVectors == 0 {
		r.mu.RUnlock()
		return nil, nil
	}

	hybridQuery := QuantizeHybrid(query, r.hybridCfg)
	vectorSize := r.vectorSize
	revMap := r.revMap
	vectors := r.vectors
	dim := r.config.FullDim
	cfg := r.hybridCfg
	numWorkers := r.config.NumWorkers
	if numWorkers > numVectors {
		numWorkers = numVectors
	}

	vectorsPerWorker := (numVectors + numWorkers - 1) / numWorkers

	// Pre-allocate results slice for all workers
	allResults := make([]SearchResult, 0, k*numWorkers)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		startIdx := i * vectorsPerWorker
		endIdx := startIdx + vectorsPerWorker
		if endIdx > numVectors {
			endIdx = numVectors
		}
		if startIdx >= endIdx {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			topK := scanChunkHybrid(vectors, hybridQuery, start, end, k, vectorSize, revMap, dim, cfg)
			mu.Lock()
			allResults = append(allResults, topK...)
			mu.Unlock()
		}(startIdx, endIdx)
	}

	wg.Wait()
	r.mu.RUnlock()
	return getTopK(allResults, k), nil
}
```

---

## 6. Add sync.Pool for QuantizeHybrid allocations

**File:** `hybrid.go` (new file `pool.go` or inline in `hybrid.go`)

**Problem:** `QuantizeHybrid` allocates a `padded := make([]float32, paddedDim)` slice on every call. For high-throughput ingestion, this creates GC pressure.

**Fix:** Create a sync.Pool for padded buffers:

```go
// pool.go
package turboquant

import "sync"

var paddedPool = sync.Pool{
	New: func() any {
		return make([]float32, 4096) // max common padded size (2048 for 1536-dim, 4096 for larger)
	},
}

func getPaddedBuffer(size int) []float32 {
	buf := paddedPool.Get().([]float32)
	if len(buf) < size {
		buf = make([]float32, size)
	}
	return buf[:size]
}

func putPaddedBuffer(buf []float32) {
	cap := cap(buf)
	if cap >= 1024 && cap <= 8192 {
		paddedPool.Put(buf[:cap])
	}
}
```

Then modify `QuantizeHybrid`:

```go
func QuantizeHybrid(vec []float32, cfg *HybridConfig) []byte {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	dim := len(vec)
	paddedDim := nextPow2(dim)

	padded := getPaddedBuffer(paddedDim)
	copy(padded, vec)
	// Zero out padding region
	for i := dim; i < paddedDim; i++ {
		padded[i] = 0
	}
	defer putPaddedBuffer(padded)

	FWHT_Optimized(padded)

	invNorm := 1.0 / float32(math.Sqrt(float64(paddedDim)))
	for i := range padded {
		padded[i] *= invNorm
	}

	blockSize := cfg.BlockSize
	numBlocks := (paddedDim + blockSize - 1) / blockSize
	outSize := HybridVectorSize(paddedDim, cfg)
	out := make([]byte, outSize)

	// ... rest unchanged ...
}
```

**Note:** The `DequantizeHybrid` also allocates `vec := make([]float32, paddedDim)`. A similar pool can be used there, but the pooled buffer must be at least `paddedDim` size.

---

## 7. Additional minor optimizations

### 7a. Use `bits.Len32` for nextPow2

**File:** `fwht.go`

```go
import "math/bits"

func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	return 1 << bits.Len32(uint32(n-1))
}
```

### 7b. Remove dead `UseLUT` field

**File:** `hybrid.go` — The `UseLUT` field in `HybridConfig` is always effectively true. Either remove it or make it functional.

### 7c. Simplify getTopK for small result sets

**File:** `search.go` — When merging worker results (typically ≤ 40 items), `sort.Slice` is faster than heap + sort:

```go
func getTopK(results []SearchResult, k int) []SearchResult {
	if len(results) <= k {
		if len(results) > 1 {
			sort.Slice(results, func(i, j int) bool {
				return results[i].Score > results[j].Score
			})
		}
		return results
	}

	// For small merges, sort.Slice is faster than heap
	if len(results) < 256 {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Score > results[j].Score
		})
		if len(results) > k {
			results = results[:k]
		}
		return results
	}

	// For large merges, use heap
	h := make(scoreHeap, k)
	heap.Init(&h)

	for i, r := range results {
		if i < k {
			h[i] = scoreIndex{score: r.Score, idx: i}
		} else if r.Score > h[0].score {
			h[0] = scoreIndex{score: r.Score, idx: i}
			heap.Fix(&h, 0)
		}
	}

	sorted := make([]SearchResult, 0, k)
	for _, si := range h {
		sorted = append(sorted, results[si.idx])
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})
	return sorted
}
```

---

## Verification Steps

After applying all changes:

1. **Run tests:** `go test -v -count=1 ./...`
2. **Run with race detector:** `go test -race -count=1 ./...`
3. **Run benchmarks:** `go test -bench=. -benchmem -count=3`
4. **Verify 4-bit dot product correctness:** The variable shadowing fix should improve accuracy for 4-bit search results — check `TestQuantizationAccuracy4Bit` passes consistently.
