//go:build !amd64 || noasm

package turboquant

func dotProductHybridBatch(query []byte, vectors []byte, numVectors int, vectorSize int, dim int, cfg *HybridConfig, scores []float32) {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	for i := 0; i < numVectors; i++ {
		scores[i] = DotProductHybrid(query, vectors[i*vectorSize:(i+1)*vectorSize], dim, cfg)
	}
}
