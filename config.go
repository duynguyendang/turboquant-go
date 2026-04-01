package turboquant

// Config holds configuration for TurboQuant vector operations.
type Config struct {
	// FullDim is the original dimensionality of input vectors (e.g., 1536 for OpenAI embeddings)
	FullDim int
	// HybridBitWidth is the quantization bit width (4 or 8)
	HybridBitWidth int
	// HybridBlockSize is the block size for block-wise quantization (must be divisible by 8)
	HybridBlockSize int
	// NumWorkers is the number of parallel workers for search operations
	NumWorkers int
	// VectorCapacity is the initial capacity for vector storage
	VectorCapacity int
	// SegmentSize is the size of each mmap segment in bytes
	SegmentSize int
}

// DefaultConfig returns a default configuration suitable for most use cases.
func DefaultConfig() *Config {
	return &Config{
		FullDim:         1536,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  100000,
		SegmentSize:     64 << 20, // 64MB
	}
}

// Validate checks if the configuration is valid.
func (c *Config) Validate() error {
	if c.FullDim <= 0 {
		return ErrInvalidDimension
	}
	if c.HybridBitWidth != 4 && c.HybridBitWidth != 8 {
		return ErrInvalidBitWidth
	}
	if c.HybridBlockSize <= 0 || c.HybridBlockSize%8 != 0 {
		return ErrInvalidBlockSize
	}
	if c.NumWorkers <= 0 {
		c.NumWorkers = 1
	}
	if c.VectorCapacity <= 0 {
		c.VectorCapacity = 10000
	}
	return nil
}

// hybridConfig converts Config to the internal HybridConfig.
func (c *Config) hybridConfig() *HybridConfig {
	return &HybridConfig{
		BitWidth:  c.HybridBitWidth,
		BlockSize: c.HybridBlockSize,
	}
}

// PaddedDim returns the dimension after padding to next power of 2.
// This is the actual dimension used for compression.
func (c *Config) PaddedDim() int {
	dim := c.FullDim
	if dim <= 1 {
		return 1
	}
	p := 1
	for p < dim {
		p <<= 1
	}
	return p
}
