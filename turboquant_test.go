package turboquant

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// Benchmark configuration
const (
	benchDim        = 1536
	benchVectorCount = 100000
	benchTopK        = 10
)

var (
	benchRegistry *Registry
	benchVectors  [][]float32
	benchQuery    []float32
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// generateRandomVector creates a random normalized vector of given dimension.
func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	var sumSquares float32
	for i := 0; i < dim; i++ {
		v := float32(rand.NormFloat64() * 0.1)
		vec[i] = v
		sumSquares += v * v
	}
	// L2 normalize
	mag := float32(math.Sqrt(float64(sumSquares)))
	if mag > 1e-10 {
		for i := range vec {
			vec[i] /= mag
		}
	}
	return vec
}

// setupBenchRegistry creates a registry with benchVectorCount random vectors.
func setupBenchRegistry(t *testing.B) *Registry {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  benchVectorCount,
	}
	reg, err := NewRegistry(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Pre-generate vectors
	vecs := make([][]float32, benchVectorCount)
	for i := 0; i < benchVectorCount; i++ {
		vecs[i] = generateRandomVector(benchDim)
	}

	// Add all vectors
	for i := 0; i < benchVectorCount; i++ {
		if err := reg.Add(uint64(i), vecs[i]); err != nil {
			t.Fatal(err)
		}
	}

	return reg
}

func setupBench(t *testing.B) {
	if benchRegistry == nil || benchRegistry.Count() == 0 {
		benchRegistry = setupBenchRegistry(t)
		benchQuery = generateRandomVector(benchDim)
	}
}

// --- Quantize Benchmark ---

func BenchmarkQuantize(b *testing.B) {
	cfg := DefaultHybridConfig()
	vec := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		QuantizeHybrid(vec, cfg)
	}
}

func BenchmarkQuantize4Bit(b *testing.B) {
	cfg := &HybridConfig{BitWidth: 4, BlockSize: 32}
	vec := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		QuantizeHybrid(vec, cfg)
	}
}

// --- Dequantize Benchmark ---

func BenchmarkDequantize(b *testing.B) {
	cfg := DefaultHybridConfig()
	vec := generateRandomVector(benchDim)
	compressed := QuantizeHybrid(vec, cfg)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		DequantizeHybrid(compressed, benchDim, cfg)
	}
}

func BenchmarkDequantize4Bit(b *testing.B) {
	cfg := &HybridConfig{BitWidth: 4, BlockSize: 32}
	vec := generateRandomVector(benchDim)
	compressed := QuantizeHybrid(vec, cfg)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		DequantizeHybrid(compressed, benchDim, cfg)
	}
}

// --- Dot Product Benchmarks ---

func BenchmarkDotProductFull(b *testing.B) {
	vecA := generateRandomVector(benchDim)
	vecB := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		DotProduct(vecA, vecB)
	}
}

func BenchmarkDotProductHybrid(b *testing.B) {
	cfg := DefaultHybridConfig()
	vecA := generateRandomVector(benchDim)
	vecB := generateRandomVector(benchDim)
	compA := QuantizeHybrid(vecA, cfg)
	compB := QuantizeHybrid(vecB, cfg)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		DotProductHybrid(compA, compB, benchDim, cfg)
	}
}

func BenchmarkDotProductHybrid4Bit(b *testing.B) {
	cfg := &HybridConfig{BitWidth: 4, BlockSize: 32}
	vecA := generateRandomVector(benchDim)
	vecB := generateRandomVector(benchDim)
	compA := QuantizeHybrid(vecA, cfg)
	compB := QuantizeHybrid(vecB, cfg)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		DotProductHybrid(compA, compB, benchDim, cfg)
	}
}

func BenchmarkDotProductHybridFull(b *testing.B) {
	cfg := DefaultHybridConfig()
	vecA := generateRandomVector(benchDim)
	vecB := generateRandomVector(benchDim)
	compA := QuantizeHybrid(vecA, cfg)
	compB := QuantizeHybrid(vecB, cfg)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		DotProductHybridFull(compA, compB, benchDim, cfg)
	}
}

// --- Search Benchmarks ---

func BenchmarkSearch(b *testing.B) {
	setupBench(b)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		benchRegistry.Search(benchQuery, benchTopK)
	}
}

func BenchmarkSearch1K(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  1000,
	}
	reg, _ := NewRegistry(cfg)
	vecs := make([][]float32, 1000)
	for i := 0; i < 1000; i++ {
		vecs[i] = generateRandomVector(benchDim)
		reg.Add(uint64(i), vecs[i])
	}
	query := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		reg.Search(query, benchTopK)
	}
}

func BenchmarkSearch10K(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  10000,
	}
	reg, _ := NewRegistry(cfg)
	vecs := make([][]float32, 10000)
	for i := 0; i < 10000; i++ {
		vecs[i] = generateRandomVector(benchDim)
		reg.Add(uint64(i), vecs[i])
	}
	query := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		reg.Search(query, benchTopK)
	}
}

func BenchmarkSearch100K(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  100000,
	}
	reg, _ := NewRegistry(cfg)
	vecs := make([][]float32, 100000)
	for i := 0; i < 100000; i++ {
		vecs[i] = generateRandomVector(benchDim)
		reg.Add(uint64(i), vecs[i])
	}
	query := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		reg.Search(query, benchTopK)
	}
}

func BenchmarkSearch500K(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  500000,
	}
	reg, _ := NewRegistry(cfg)
	vecs := make([][]float32, 500000)
	for i := 0; i < 500000; i++ {
		vecs[i] = generateRandomVector(benchDim)
		reg.Add(uint64(i), vecs[i])
	}
	query := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		reg.Search(query, benchTopK)
	}
}

func BenchmarkSearch4Bit(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  4,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  benchVectorCount,
	}
	reg, _ := NewRegistry(cfg)
	vecs := make([][]float32, benchVectorCount)
	for i := 0; i < benchVectorCount; i++ {
		vecs[i] = generateRandomVector(benchDim)
		reg.Add(uint64(i), vecs[i])
	}
	query := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		reg.Search(query, benchTopK)
	}
}

// --- Registry Benchmarks ---

func BenchmarkRegistryAdd(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      1,
		VectorCapacity:  b.N,
	}
	reg, _ := NewRegistry(cfg)
	vec := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		reg.Add(uint64(i), vec)
	}
}

func BenchmarkRegistryGet(b *testing.B) {
	setupBench(b)

	b.ResetTimer()
	b.ReportAllocs()

	id := uint64(rand.Intn(benchVectorCount))
	for i := 0; i < b.N; i++ {
		benchRegistry.Get(id)
	}
}

func BenchmarkRegistryGetDecompressed(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      1,
		VectorCapacity:  1000,
	}
	reg, _ := NewRegistry(cfg)
	for i := 0; i < 1000; i++ {
		reg.Add(uint64(i), generateRandomVector(benchDim))
	}

	b.ResetTimer()
	b.ReportAllocs()

	id := uint64(500)
	for i := 0; i < b.N; i++ {
		reg.GetDecompressed(id)
	}
}

// --- FWHT Benchmarks ---

func BenchmarkFWHT(b *testing.B) {
	vec := generateRandomVector(2048) // 1536 rounds up to 2048

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		FWHT(vec)
	}
}

func BenchmarkFWHT1024(b *testing.B) {
	vec := generateRandomVector(1024)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		FWHT(vec)
	}
}

func BenchmarkFWHT2048(b *testing.B) {
	vec := generateRandomVector(2048)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		FWHT(vec)
	}
}

func BenchmarkFWHT4096(b *testing.B) {
	vec := generateRandomVector(4096)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		FWHT(vec)
	}
}

// --- Memory Usage Benchmarks ---

func BenchmarkMemoryCompressedSize(b *testing.B) {
	cfg := DefaultHybridConfig()
	vec := generateRandomVector(benchDim)
	compressed := QuantizeHybrid(vec, cfg)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = len(compressed)
	}
}

func BenchmarkMemoryRegistry(b *testing.B) {
	cfg := &Config{
		FullDim:         benchDim,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  benchVectorCount,
	}
	reg, _ := NewRegistry(cfg)
	vec := generateRandomVector(benchDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < benchVectorCount; i++ {
		reg.Add(uint64(i), vec)
	}

	_ = reg.Count()
}

// --- Accuracy Benchmarks ---

func TestQuantizationAccuracy8Bit(t *testing.T) {
	cfg := DefaultHybridConfig()

	// Test multiple vectors
	for seed := 0; seed < 100; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))
		vec := make([]float32, benchDim)
		var sumSquares float32
		for i := range vec {
			vec[i] = float32(rng.NormFloat64() * 0.1)
			sumSquares += vec[i] * vec[i]
		}
		// Normalize
		mag := float32(math.Sqrt(float64(sumSquares)))
		if mag > 1e-10 {
			for i := range vec {
				vec[i] /= mag
			}
		}

		compressed := QuantizeHybrid(vec, cfg)
		decompressed := DequantizeHybrid(compressed, benchDim, cfg)

		// Compute cosine similarity
		dot := DotProduct(vec, decompressed)

		if dot < 0.999 {
			t.Errorf("8-bit accuracy too low: %.6f (seed=%d)", dot, seed)
		}
	}
}

func TestQuantizationAccuracy4Bit(t *testing.T) {
	cfg := &HybridConfig{BitWidth: 4, BlockSize: 32}

	for seed := 0; seed < 100; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))
		vec := make([]float32, benchDim)
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

		compressed := QuantizeHybrid(vec, cfg)
		decompressed := DequantizeHybrid(compressed, benchDim, cfg)

		dot := DotProduct(vec, decompressed)

		if dot < 0.99 {
			t.Errorf("4-bit accuracy too low: %.6f (seed=%d)", dot, seed)
		}
	}
}

func TestCompressedDotProductAccuracy(t *testing.T) {
	cfg := DefaultHybridConfig()

	for seed := 0; seed < 50; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))
		vecA := make([]float32, benchDim)
		vecB := make([]float32, benchDim)
		for i := range vecA {
			vecA[i] = float32(rng.NormFloat64() * 0.1)
			vecB[i] = float32(rng.NormFloat64() * 0.1)
		}
		vecA = L2Normalize(vecA)
		vecB = L2Normalize(vecB)

		compA := QuantizeHybrid(vecA, cfg)
		compB := QuantizeHybrid(vecB, cfg)

		fullDot := DotProduct(vecA, vecB)
		compressedDot := DotProductHybrid(compA, compB, benchDim, cfg)

		diff := math.Abs(float64(fullDot - compressedDot))
		if diff > 0.001 {
			t.Errorf("Compressed dot product mismatch: full=%.6f, compressed=%.6f, diff=%.6f",
				fullDot, compressedDot, diff)
		}
	}
}

// --- Example/Print functions for manual verification ---

func ExampleQuantizeHybrid() {
	cfg := DefaultHybridConfig()
	vec := generateRandomVector(1536)

	compressed := QuantizeHybrid(vec, cfg)
	decompressed := DequantizeHybrid(compressed, 1536, cfg)

	// Compute compression ratio
	originalSize := 1536 * 4 // float32 = 4 bytes
	compressedSize := len(compressed)
	ratio := float64(compressedSize) / float64(originalSize)

	fmt.Printf("Original: %d bytes\n", originalSize)
	fmt.Printf("Compressed: %d bytes\n", compressedSize)
	fmt.Printf("Compression ratio: %.2f%%\n", ratio*100)
	fmt.Printf("Reconstruction similarity: %.6f\n", DotProduct(vec, decompressed))
}

func ExampleRegistry() {
	reg, _ := NewRegistry(nil)

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := generateRandomVector(1536)
		reg.Add(uint64(i), vec)
	}

	// Search
	query := generateRandomVector(1536)
	results, _ := reg.Search(query, 3)

	fmt.Printf("Found %d results\n", len(results))
	for _, r := range results {
		fmt.Printf("  ID: %d, Score: %.4f\n", r.ID, r.Score)
	}
}

func PrintSearchScale() {
	sizes := []int{1000, 10000, 100000}

	for _, n := range sizes {
		cfg := &Config{
			FullDim:         1536,
			HybridBitWidth:  8,
			HybridBlockSize: 32,
			NumWorkers:      4,
			VectorCapacity:  n,
		}
		reg, _ := NewRegistry(cfg)

		start := time.Now()
		for i := 0; i < n; i++ {
			reg.Add(uint64(i), generateRandomVector(1536))
		}
		addDuration := time.Since(start)

		query := generateRandomVector(1536)
		start = time.Now()
		results, _ := reg.Search(query, 10)
		searchDuration := time.Since(start)

		fmt.Printf("n=%d: Add=%.2fms, Search=%.2fms, Results=%d\n",
			n, float64(addDuration.Milliseconds()), float64(searchDuration.Milliseconds()), len(results))
	}
}
