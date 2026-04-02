package turboquant

import "errors"

// Config holds configuration for TurboQuant vector operations.
type Config struct {
	FullDim         int
	HybridBitWidth  int
	HybridBlockSize int
	NumWorkers      int
	VectorCapacity  int
	EnableQJL       bool
}

// DefaultConfig returns a default configuration suitable for most use cases.
func DefaultConfig() *Config {
	return &Config{
		FullDim:         1536,
		HybridBitWidth:  8,
		HybridBlockSize: 32,
		NumWorkers:      4,
		VectorCapacity:  100000,
		EnableQJL:       false,
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
		return errors.New("invalid num workers: must be positive")
	}
	if c.VectorCapacity <= 0 {
		return errors.New("invalid vector capacity: must be positive")
	}
	return nil
}

// hybridConfig converts Config to the internal HybridConfig.
func (c *Config) hybridConfig() *HybridConfig {
	return &HybridConfig{
		BitWidth:  c.HybridBitWidth,
		BlockSize: c.HybridBlockSize,
		EnableQJL: c.EnableQJL,
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
