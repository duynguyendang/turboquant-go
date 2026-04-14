//go:build amd64 && !noasm
// +build amd64,!noasm

package turboquant

// FWHT_SIMD performs the in-place Fast Walsh-Hadamard Transform
// using optimized butterfly operations with multiple accumulators.
//
//go:noinline
func FWHT_SIMD(vec []float32) {
	n := len(vec)

	// Process 4 butterfly operations at a time for better pipelining
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			end := i + h
			j := i
			// Process 4 butterfly operations at a time
			for ; j+4 <= end; j += 4 {
				x0, x1, x2, x3 := vec[j], vec[j+1], vec[j+2], vec[j+3]
				y0, y1, y2, y3 := vec[j+h], vec[j+h+1], vec[j+h+2], vec[j+h+3]

				vec[j] = x0 + y0
				vec[j+1] = x1 + y1
				vec[j+2] = x2 + y2
				vec[j+3] = x3 + y3

				vec[j+h] = x0 - y0
				vec[j+h+1] = x1 - y1
				vec[j+h+2] = x2 - y2
				vec[j+h+3] = x3 - y3
			}
			// Handle remaining
			for ; j < end; j++ {
				x := vec[j]
				y := vec[j+h]
				vec[j] = x + y
				vec[j+h] = x - y
			}
		}
	}
}

func init() {
	FWHT_Optimized = FWHT_SIMD
}
