package turboquant

import "math"

// L2Normalize normalizes a vector to unit length.
func L2Normalize(vec []float32) []float32 {
	var sumSquares float32
	for _, v := range vec {
		sumSquares += v * v
	}

	magnitude := float32(math.Sqrt(float64(sumSquares)))

	result := make([]float32, len(vec))
	if magnitude < 1e-10 {
		return result
	}

	invMag := 1.0 / magnitude
	for i, v := range vec {
		result[i] = v * invMag
	}

	return result
}

// DotProduct computes the dot product of two vectors.
func DotProduct(v1, v2 []float32) float32 {
	if len(v1) != len(v2) {
		panic("DotProduct: dimension mismatch")
	}

	var sum float32
	for i := range v1 {
		sum += v1[i] * v2[i]
	}

	return sum
}

// CosineSimilarity computes the cosine similarity between two vectors.
// For L2-normalized vectors, this equals the dot product.
func CosineSimilarity(v1, v2 []float32) float32 {
	return DotProduct(v1, v2)
}
