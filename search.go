package turboquant

import (
	"container/heap"
	"iter"
	"sort"
)

// SearchResult represents a search result with ID and score.
type SearchResult struct {
	ID    uint64
	Score float32
}

type scoreIndex struct {
	score float32
	idx   int
}

type scoreHeap []scoreIndex

func (h scoreHeap) Len() int           { return len(h) }
func (h scoreHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h scoreHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *scoreHeap) Push(x any)        { *h = append(*h, x.(scoreIndex)) }
func (h *scoreHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// Search finds the top-k most similar vectors to the query.
func (r *Registry) Search(query []float32, k int) ([]SearchResult, error) {
	return r.SearchWithLimit(query, k)
}

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

	type result struct {
		results []SearchResult
	}
	resultCh := make(chan result, numWorkers)

	for i := 0; i < numWorkers; i++ {
		startIdx := i * vectorsPerWorker
		endIdx := startIdx + vectorsPerWorker
		if endIdx > numVectors {
			endIdx = numVectors
		}
		if startIdx >= endIdx {
			continue
		}

		go func(start, end int) {
			topK := scanChunkHybrid(vectors, hybridQuery, start, end, k, vectorSize, revMap, dim, cfg)
			resultCh <- result{results: topK}
		}(startIdx, endIdx)
	}

	allResults := make([]SearchResult, 0, k*numWorkers)
	for i := 0; i < numWorkers; i++ {
		res := <-resultCh
		allResults = append(allResults, res.results...)
	}

	r.mu.RUnlock()
	return getTopK(allResults, k), nil
}

// scanChunkHybrid scans a chunk of vectors and returns the top-k results.
func scanChunkHybrid(vectors []byte, query []byte, startIdx, endIdx, k int, vectorSize int, revMap []uint64, dim int, cfg *HybridConfig) []SearchResult {
	h := make(scoreHeap, 0, k)
	numVectors := endIdx - startIdx
	scores := make([]float32, numVectors)

	dotProductHybridBatch(query, vectors[startIdx*vectorSize:endIdx*vectorSize], numVectors, vectorSize, dim, cfg, scores)

	for idx := 0; idx < numVectors; idx++ {
		score := scores[idx]

		if len(h) < k {
			h = append(h, scoreIndex{score: score, idx: startIdx + idx})
			if len(h) == k {
				heap.Init(&h)
			}
		} else if score > h[0].score {
			h[0] = scoreIndex{score: score, idx: startIdx + idx}
			heap.Fix(&h, 0)
		}
	}

	results := make([]SearchResult, 0, len(h))
	for _, si := range h {
		if si.idx < len(revMap) {
			results = append(results, SearchResult{
				ID:    revMap[si.idx],
				Score: si.score,
			})
		}
	}

	return results
}

func getTopK(results []SearchResult, k int) []SearchResult {
	if len(results) <= k {
		if len(results) > 1 {
			sort.Slice(results, func(i, j int) bool {
				return results[i].Score > results[j].Score
			})
		}
		return results
	}

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
	vectors := make([]byte, len(r.vectors))
	copy(vectors, r.vectors)
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
