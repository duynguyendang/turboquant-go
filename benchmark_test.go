package turboquant

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"testing"
	"time"
)

// ============================================================
// Helpers
// ============================================================

func generateDeterministicVector(dim int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	vec := make([]float32, dim)
	var sumSquares float32
	for i := range vec {
		vec[i] = float32(rng.NormFloat64() * 0.1)
		sumSquares += vec[i] * vec[i]
	}
	mag := float32(math.Sqrt(float64(sumSquares)))
	if mag > 1e-10 {
		for i := range vec {
			vec[i] /= mag
		}
	}
	return vec
}

func buildRegistry(n int, dim int, cfg *Config, seed int64) *Registry {
	reg, err := NewRegistry(cfg)
	if err != nil {
		panic(err)
	}
	for i := 0; i < n; i++ {
		vec := generateDeterministicVector(dim, seed+int64(i))
		reg.Add(uint64(i), vec)
	}
	return reg
}

// ============================================================
// Clustered Vector Generation
// ============================================================

var (
	clusterCenterCache sync.Map // key: "dim:numClusters:clusterID"
)

func getClusterCenter(dim, numClusters, clusterID int) []float32 {
	key := fmt.Sprintf("%d:%d:%d", dim, numClusters, clusterID)
	if v, ok := clusterCenterCache.Load(key); ok {
		return v.([]float32)
	}
	rng := rand.New(rand.NewSource(int64(clusterID * 1000003)))
	center := make([]float32, dim)
	var sumSquares float32
	for i := range center {
		center[i] = float32(rng.NormFloat64() * 0.5)
		sumSquares += center[i] * center[i]
	}
	mag := float32(math.Sqrt(float64(sumSquares)))
	if mag > 1e-10 {
		for i := range center {
			center[i] /= mag
		}
	}
	clusterCenterCache.Store(key, center)
	return center
}

func generateClusteredVector(dim, numClusters, clusterID int, spread float64, rng *rand.Rand) []float32 {
	center := getClusterCenter(dim, numClusters, clusterID)
	vec := make([]float32, dim)
	var sumSquares float32
	for i := range vec {
		vec[i] = center[i] + float32(rng.NormFloat64()*spread)
		sumSquares += vec[i] * vec[i]
	}
	mag := float32(math.Sqrt(float64(sumSquares)))
	if mag > 1e-10 {
		for i := range vec {
			vec[i] /= mag
		}
	}
	return vec
}

func buildClusteredRegistry(n, dim, numClusters int, spread float64, cfg *Config, seed int64) *Registry {
	reg, err := NewRegistry(cfg)
	if err != nil {
		panic(err)
	}
	vecsPerCluster := n / numClusters
	idx := 0
	for c := 0; c < numClusters; c++ {
		for j := 0; j < vecsPerCluster; j++ {
			rng := rand.New(rand.NewSource(seed + int64(idx)))
			vec := generateClusteredVector(dim, numClusters, c, spread, rng)
			reg.Add(uint64(idx), vec)
			idx++
		}
	}
	return reg
}

func fp32GroundTruth(reg *Registry, query []float32, k int) []SearchResult {
	r := reg
	r.mu.RLock()
	numVectors := r.totalVectors
	revMap := make([]uint64, len(r.revMap))
	copy(revMap, r.revMap)
	vectors := r.vectors
	vectorSize := r.vectorSize
	dim := r.config.FullDim
	cfg := r.hybridCfg
	r.mu.RUnlock()

	type scored struct {
		id    uint64
		score float32
	}
	scores := make([]scored, 0, numVectors)
	for i := 0; i < numVectors; i++ {
		vecData := vectors[i*vectorSize : (i+1)*vectorSize]
		v := DequantizeHybrid(vecData, dim, cfg)
		s := DotProduct(query, v)
		scores = append(scores, scored{revMap[i], s})
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	if k > len(scores) {
		k = len(scores)
	}
	results := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		results[i] = SearchResult{ID: scores[i].id, Score: scores[i].score}
	}
	return results
}

func recallAtK(results []SearchResult, groundTruth []SearchResult, k int) float64 {
	if len(groundTruth) < k {
		k = len(groundTruth)
	}
	if len(results) < k {
		k = len(results)
	}
	if k == 0 {
		return 1.0
	}
	gtSet := make(map[uint64]bool, k)
	for _, r := range groundTruth[:k] {
		gtSet[r.ID] = true
	}
	hits := 0
	for _, r := range results[:k] {
		if gtSet[r.ID] {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func mseScores(reg *Registry, query []float32, cfg *HybridConfig) float64 {
	r := reg
	r.mu.RLock()
	numVectors := r.totalVectors
	vectors := r.vectors
	vectorSize := r.vectorSize
	dim := r.config.FullDim
	r.mu.RUnlock()

	compQ := QuantizeHybrid(query, cfg)

	var sumSq float64
	for i := 0; i < numVectors; i++ {
		vecData := vectors[i*vectorSize : (i+1)*vectorSize]
		fp32Vec := DequantizeHybrid(vecData, dim, r.hybridCfg)
		fullDot := DotProduct(query, fp32Vec)
		compressedDot := DotProductHybrid(compQ, vecData, dim, cfg)
		diff := float64(fullDot - compressedDot)
		sumSq += diff * diff
	}
	return sumSq / float64(numVectors)
}

// ============================================================
// Section 1: Micro-Benchmarks
// ============================================================

func BenchmarkFWHT_Compare(b *testing.B) {
	dims := []int{1024, 2048, 4096}
	for _, dim := range dims {
		vec := generateRandomVector(dim)
		b.Run(fmt.Sprintf("Vanilla/%d", dim), func(b *testing.B) {
			v := make([]float32, dim)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(v, vec)
				FWHT(v)
			}
		})
		b.Run(fmt.Sprintf("Optimized/%d", dim), func(b *testing.B) {
			v := make([]float32, dim)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(v, vec)
				FWHT_Optimized(v)
			}
		})
	}
}

func BenchmarkQuantize_AllModes(b *testing.B) {
	vec := generateRandomVector(benchDim)
	modes := []struct {
		name string
		cfg  *HybridConfig
	}{
		{"8bit", DefaultHybridConfig()},
		{"4bit", &HybridConfig{BitWidth: 4, BlockSize: 32, UseLUT: true}},
		{"4bit+QJL", &HybridConfig{BitWidth: 4, BlockSize: 32, UseLUT: true, EnableQJL: true}},
	}
	for _, m := range modes {
		b.Run(m.name, func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				QuantizeHybrid(vec, m.cfg)
			}
		})
	}
}

func BenchmarkDequantize_AllModes(b *testing.B) {
	vec := generateRandomVector(benchDim)
	modes := []struct {
		name string
		cfg  *HybridConfig
	}{
		{"8bit", DefaultHybridConfig()},
		{"4bit", &HybridConfig{BitWidth: 4, BlockSize: 32, UseLUT: true}},
		{"4bit+QJL", &HybridConfig{BitWidth: 4, BlockSize: 32, UseLUT: true, EnableQJL: true}},
	}
	for _, m := range modes {
		comp := QuantizeHybrid(vec, m.cfg)
		b.Run(m.name, func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(benchDim * 4))
			for i := 0; i < b.N; i++ {
				DequantizeHybrid(comp, benchDim, m.cfg)
			}
		})
	}
}

func BenchmarkDotProduct_Shootout(b *testing.B) {
	vecA := generateRandomVector(benchDim)
	vecB := generateRandomVector(benchDim)

	cfg8 := DefaultHybridConfig()
	cfg4 := &HybridConfig{BitWidth: 4, BlockSize: 32, UseLUT: true}
	cfg4q := &HybridConfig{BitWidth: 4, BlockSize: 32, UseLUT: true, EnableQJL: true}

	compA8 := QuantizeHybrid(vecA, cfg8)
	compB8 := QuantizeHybrid(vecB, cfg8)
	compA4 := QuantizeHybrid(vecA, cfg4)
	compB4 := QuantizeHybrid(vecB, cfg4)
	compA4q := QuantizeHybrid(vecA, cfg4q)
	compB4q := QuantizeHybrid(vecB, cfg4q)

	b.Run("FP32", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProduct(vecA, vecB)
		}
	})
	b.Run("8bit", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProductHybrid(compA8, compB8, benchDim, cfg8)
		}
	})
	b.Run("4bit", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProductHybrid(compA4, compB4, benchDim, cfg4)
		}
	})
	b.Run("4bit+QJL", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProductHybrid(compA4q, compB4q, benchDim, cfg4q)
		}
	})
	b.Run("HybridFull", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			DotProductHybridFull(compA8, compB8, benchDim, cfg8)
		}
	})
}

func BenchmarkQuantize_DimScaling(b *testing.B) {
	dims := []int{256, 512, 1024, 1536, 2048, 4096}
	cfg := &HybridConfig{BitWidth: 4, BlockSize: 32, UseLUT: true}
	for _, dim := range dims {
		vec := generateRandomVector(dim)
		b.Run(fmt.Sprintf("%d", dim), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				QuantizeHybrid(vec, cfg)
			}
		})
	}
}

// ============================================================
// Section 2: Search Benchmarks
// ============================================================

func BenchmarkSearch_LatencyScaling(b *testing.B) {
	sizes := []int{10000, 100000, 500000}
	modes := []struct {
		name string
		cfg  *Config
	}{
		{"8bit", &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4}},
		{"4bit", &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4}},
		{"4bit+QJL", &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, EnableQJL: true}},
	}
	for _, size := range sizes {
		for _, m := range modes {
			m.cfg.VectorCapacity = size
			reg := buildRegistry(size, benchDim, m.cfg, 42)
			query := generateDeterministicVector(benchDim, 999)
			b.Run(fmt.Sprintf("%s/%d", m.name, size), func(b *testing.B) {
				b.ResetTimer()
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					reg.Search(query, 10)
				}
			})
		}
	}
}

func BenchmarkSearch_QPS(b *testing.B) {
	cfg := &Config{
		FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32,
		NumWorkers: 4, VectorCapacity: 100000,
	}
	reg := buildRegistry(100000, benchDim, cfg, 42)
	query := generateDeterministicVector(benchDim, 999)

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			reg.Search(query, 10)
		}
	})
}

func BenchmarkSearch_Parallelism(b *testing.B) {
	workers := []int{1, 4, 8, 12}
	cfg := &Config{
		FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32,
		NumWorkers: 4, VectorCapacity: 100000,
	}
	reg := buildRegistry(100000, benchDim, cfg, 42)
	query := generateDeterministicVector(benchDim, 999)

	for _, w := range workers {
		cfg := *cfg
		cfg.NumWorkers = w
		reg2, _ := NewRegistry(&cfg)
		for i := 0; i < 100000; i++ {
			vec := generateDeterministicVector(benchDim, int64(i)+42)
			reg2.Add(uint64(i), vec)
		}
		b.Run(fmt.Sprintf("workers=%d", w), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				reg2.Search(query, 10)
			}
		})
	}
	_ = reg
}

func BenchmarkSearch_ColdVsWarm(b *testing.B) {
	cfg := &Config{
		FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32,
		NumWorkers: 4, VectorCapacity: 100000,
	}
	reg := buildRegistry(100000, benchDim, cfg, 42)
	query := generateDeterministicVector(benchDim, 999)

	b.Run("Cold", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			cfg2 := *cfg
			cfg2.VectorCapacity = 100000
			reg2 := buildRegistry(100000, benchDim, &cfg2, int64(i*100))
			b.StartTimer()
			reg2.Search(query, 10)
			b.StopTimer()
		}
	})

	b.Run("Warm", func(b *testing.B) {
		reg.Search(query, 10)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			reg.Search(query, 10)
		}
	})
}

func BenchmarkSearch_TopKScaling(b *testing.B) {
	ks := []int{1, 5, 10, 50, 100}
	cfg := &Config{
		FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32,
		NumWorkers: 4, VectorCapacity: 100000,
	}
	reg := buildRegistry(100000, benchDim, cfg, 42)
	query := generateDeterministicVector(benchDim, 999)

	for _, k := range ks {
		b.Run(fmt.Sprintf("K=%d", k), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				reg.Search(query, k)
			}
		})
	}
}

// ============================================================
// Section 3: Accuracy Tests
// ============================================================

func TestRecallAtK(t *testing.T) {
	n := 10000
	querySeed := int64(42)

	cfg8 := &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

	reg8 := buildRegistry(n, benchDim, cfg8, 42)
	reg4 := buildRegistry(n, benchDim, cfg4, 42)
	reg4q := buildRegistry(n, benchDim, cfg4q, 42)

	query := generateDeterministicVector(benchDim, querySeed)
	gt := fp32GroundTruth(reg8, query, 100)

	modes := []struct {
		name string
		reg  *Registry
		cfg  *HybridConfig
	}{
		{"8bit", reg8, cfg8.hybridConfig()},
		{"4bit", reg4, cfg4.hybridConfig()},
		{"4bit+QJL", reg4q, cfg4q.hybridConfig()},
	}

	for _, m := range modes {
		results, _ := m.reg.Search(query, 100)
		for _, k := range []int{1, 10, 100} {
			r := recallAtK(results, gt, k)
			t.Logf("Recall@%d (%s): %.4f", k, m.name, r)
		}
	}
}

func TestMSE_Scores(t *testing.T) {
	n := 1000
	cfg8 := &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

	reg8 := buildRegistry(n, benchDim, cfg8, 42)
	reg4 := buildRegistry(n, benchDim, cfg4, 42)
	reg4q := buildRegistry(n, benchDim, cfg4q, 42)

	query := generateDeterministicVector(benchDim, 999)

	mse8 := mseScores(reg8, query, cfg8.hybridConfig())
	mse4 := mseScores(reg4, query, cfg4.hybridConfig())
	mse4q := mseScores(reg4q, query, cfg4q.hybridConfig())

	t.Logf("MSE 8-bit:    %.8f", mse8)
	t.Logf("MSE 4-bit:    %.8f", mse4)
	t.Logf("MSE 4-bit+QJL: %.8f", mse4q)

	if mse4q > mse4*1.5 {
		t.Errorf("QJL MSE should not be significantly worse than plain 4-bit")
	}
}

func BenchmarkQJL_Uplift(b *testing.B) {
	n := 100000
	cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

	reg4 := buildRegistry(n, benchDim, cfg4, 42)
	reg4q := buildRegistry(n, benchDim, cfg4q, 42)

	query := generateDeterministicVector(benchDim, 999)
	gt := fp32GroundTruth(reg4, query, 100)

	results4, _ := reg4.Search(query, 100)
	results4q, _ := reg4q.Search(query, 100)

	r4 := recallAtK(results4, gt, 10)
	r4q := recallAtK(results4q, gt, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.ReportMetric(r4, "recall_plain")
		b.ReportMetric(r4q, "recall_qjl")
		b.ReportMetric(r4q-r4, "uplift")
	}
}

func TestRecall_BySimilarity(t *testing.T) {
	n := 10000
	cfg8 := &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

	reg8 := buildRegistry(n, benchDim, cfg8, 42)
	reg4 := buildRegistry(n, benchDim, cfg4, 42)
	reg4q := buildRegistry(n, benchDim, cfg4q, 42)

	type bucket struct {
		label  string
		lo, hi float32
	}
	buckets := []bucket{
		{"0.5-0.6", 0.5, 0.6},
		{"0.6-0.7", 0.6, 0.7},
		{"0.7-0.8", 0.7, 0.8},
		{"0.8-0.9", 0.8, 0.9},
		{"0.9-1.0", 0.9, 1.0},
	}

	modes := []struct {
		name string
		reg  *Registry
		cfg  *HybridConfig
	}{
		{"8bit", reg8, cfg8.hybridConfig()},
		{"4bit", reg4, cfg4.hybridConfig()},
		{"4bit+QJL", reg4q, cfg4q.hybridConfig()},
	}

	for _, bucket := range buckets {
		query := generateDeterministicVector(benchDim, 42)
		gt := fp32GroundTruth(reg8, query, 100)

		for _, m := range modes {
			results, _ := m.reg.Search(query, 100)
			r := recallAtK(results, gt, 10)
			t.Logf("Recall@10 %s [%s]: %.4f (ground truth top-1 score: %.4f)", m.name, bucket.label, r, gt[0].Score)
		}
	}
}

// ============================================================
// Clustered Vector Tests
// ============================================================

func TestRecallAtK_Clustered(t *testing.T) {
	n := 10000
	numClusters := 100
	spreads := []float64{0.01, 0.005, 0.002}

	for _, spread := range spreads {
		t.Run(fmt.Sprintf("spread=%.4f", spread), func(t *testing.T) {
			cfg8 := &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
			cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
			cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

			reg8 := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg8, 42)
			reg4 := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4, 42)
			reg4q := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4q, 42)

			rng := rand.New(rand.NewSource(999))
			clusterID := rng.Intn(numClusters)
			query := generateClusteredVector(benchDim, numClusters, clusterID, spread, rand.New(rand.NewSource(9999)))
			gt := fp32GroundTruth(reg8, query, 100)

			modes := []struct {
				name string
				reg  *Registry
				cfg  *HybridConfig
			}{
				{"8bit", reg8, cfg8.hybridConfig()},
				{"4bit", reg4, cfg4.hybridConfig()},
				{"4bit+QJL", reg4q, cfg4q.hybridConfig()},
			}

			for _, m := range modes {
				results, _ := m.reg.Search(query, 100)
				for _, k := range []int{1, 10, 100} {
					r := recallAtK(results, gt, k)
					t.Logf("Recall@%d (%s): %.4f", k, m.name, r)
				}
			}
		})
	}
}

func TestMSE_Scores_Clustered(t *testing.T) {
	n := 1000
	numClusters := 100
	spreads := []float64{0.01, 0.005, 0.002}

	for _, spread := range spreads {
		t.Run(fmt.Sprintf("spread=%.4f", spread), func(t *testing.T) {
			cfg8 := &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
			cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
			cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

			reg8 := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg8, 42)
			reg4 := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4, 42)
			reg4q := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4q, 42)

			query := generateClusteredVector(benchDim, numClusters, 0, spread, rand.New(rand.NewSource(999)))

			mse8 := mseScores(reg8, query, cfg8.hybridConfig())
			mse4 := mseScores(reg4, query, cfg4.hybridConfig())
			mse4q := mseScores(reg4q, query, cfg4q.hybridConfig())

			t.Logf("MSE 8-bit:    %.8f", mse8)
			t.Logf("MSE 4-bit:    %.8f", mse4)
			t.Logf("MSE 4-bit+QJL: %.8f", mse4q)
		})
	}
}

func BenchmarkQJL_Uplift_Clustered(b *testing.B) {
	n := 100000
	numClusters := 100
	spread := 0.002

	cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

	reg4 := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4, 42)
	reg4q := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4q, 42)

	query := generateClusteredVector(benchDim, numClusters, 0, spread, rand.New(rand.NewSource(999)))
	gt := fp32GroundTruth(reg4, query, 100)

	results4, _ := reg4.Search(query, 100)
	results4q, _ := reg4q.Search(query, 100)

	r4 := recallAtK(results4, gt, 10)
	r4q := recallAtK(results4q, gt, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.ReportMetric(r4, "recall_plain")
		b.ReportMetric(r4q, "recall_qjl")
		b.ReportMetric(r4q-r4, "uplift")
	}
}

func TestRecall_BySimilarity_Clustered(t *testing.T) {
	n := 10000
	numClusters := 100
	spread := 0.002

	cfg8 := &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4 := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n}
	cfg4q := &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, VectorCapacity: n, EnableQJL: true}

	reg8 := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg8, 42)
	reg4 := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4, 42)
	reg4q := buildClusteredRegistry(n, benchDim, numClusters, spread, cfg4q, 42)

	modes := []struct {
		name string
		reg  *Registry
		cfg  *HybridConfig
	}{
		{"8bit", reg8, cfg8.hybridConfig()},
		{"4bit", reg4, cfg4.hybridConfig()},
		{"4bit+QJL", reg4q, cfg4q.hybridConfig()},
	}

	for clusterID := 0; clusterID < 5; clusterID++ {
		query := generateClusteredVector(benchDim, numClusters, clusterID, spread, rand.New(rand.NewSource(int64(clusterID*1000+999))))
		gt := fp32GroundTruth(reg8, query, 100)

		t.Logf("Query cluster %d: ground truth top-1 score = %.4f", clusterID, gt[0].Score)
		for _, m := range modes {
			results, _ := m.reg.Search(query, 100)
			r := recallAtK(results, gt, 10)
			t.Logf("  Recall@10 %s: %.4f", m.name, r)
		}
	}
}

// ============================================================
// Section 4: Resource & Memory Benchmarks
// ============================================================

func TestMemoryFootprint(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	sizes := []int{100000}
	modes := []struct {
		name string
		cfg  *Config
	}{
		{"FP32", &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4}},
		{"8bit", &Config{FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32, NumWorkers: 4}},
		{"4bit", &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4}},
		{"4bit+QJL", &Config{FullDim: benchDim, HybridBitWidth: 4, HybridBlockSize: 32, NumWorkers: 4, EnableQJL: true}},
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("N=%d", size), func(t *testing.T) {
			for _, m := range modes {
				debug.FreeOSMemory()
				var before runtime.MemStats
				runtime.ReadMemStats(&before)

				m.cfg.VectorCapacity = size
				reg := buildRegistry(size, benchDim, m.cfg, 42)

				var after runtime.MemStats
				runtime.ReadMemStats(&after)

				used := after.TotalAlloc - before.TotalAlloc
				perVector := float64(used) / float64(size)
				vecSize := reg.VectorSize()

				t.Logf("%-12s: total=%.2f MB, per-vector=%.0f B, compressed=%d B, overhead=%.0f%%",
					m.name,
					float64(used)/1024/1024,
					perVector,
					vecSize,
					(perVector-float64(vecSize))/float64(vecSize)*100,
				)
			}
		})
	}
}

func BenchmarkAllocsPerQuery(b *testing.B) {
	cfg := &Config{
		FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32,
		NumWorkers: 4, VectorCapacity: 10000,
	}
	reg := buildRegistry(10000, benchDim, cfg, 42)
	query := generateDeterministicVector(benchDim, 999)

	b.Run("Search", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			reg.Search(query, 10)
		}
	})

	b.Run("SearchIter", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for range reg.SearchIter(query, 10) {
			}
		}
	})

	b.Run("All", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for range reg.All() {
			}
		}
	})

	b.Run("IDsIter", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for range reg.IDsIter() {
			}
		}
	})
}

func TestGCPressure(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	n := 100000
	cfg := &Config{
		FullDim: benchDim, HybridBitWidth: 8, HybridBlockSize: 32,
		NumWorkers: 4, VectorCapacity: n,
	}

	debug.FreeOSMemory()
	runtime.GC()

	reg := buildRegistry(n, benchDim, cfg, 42)
	query := generateDeterministicVector(benchDim, 999)

	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	const numQueries = 100
	start := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < numQueries; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reg.Search(query, 10)
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)

	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	t.Logf("GC pressure test (%d vectors, %d queries):", n, numQueries)
	t.Logf("  Total time: %v", elapsed)
	t.Logf("  QPS: %.0f", float64(numQueries)/elapsed.Seconds())
	t.Logf("  GC runs during test: %d", after.NumGC-before.NumGC)
	t.Logf("  Heap allocated: %.2f MB", float64(after.TotalAlloc-before.TotalAlloc)/1024/1024)
	t.Logf("  Heap in use: %.2f MB", float64(after.HeapInuse)/1024/1024)
	t.Logf("  Heap objects: %d", after.HeapObjects-before.HeapObjects)
}
