package turboquant

import (
	"sync"
)

// Registry stores compressed vectors in memory and provides search functionality.
type Registry struct {
	config     *Config
	hybridCfg  *HybridConfig
	vectorSize int

	vectors    []byte // contiguous memory for all compressed vectors
	idMap      map[uint64]int
	revMap     []uint64
	totalVectors int

	mu sync.RWMutex
}

// NewRegistry creates a new vector registry with the given configuration.
func NewRegistry(cfg *Config) (*Registry, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	hybridCfg := cfg.hybridConfig()
	vectorSize := HybridVectorSize(cfg.PaddedDim(), hybridCfg)

	r := &Registry{
		config:      cfg,
		hybridCfg:   hybridCfg,
		vectorSize:  vectorSize,
		idMap:       make(map[uint64]int, cfg.VectorCapacity),
		revMap:      make([]uint64, 0, cfg.VectorCapacity),
	}

	// Pre-allocate vector storage
	r.vectors = make([]byte, 0, cfg.VectorCapacity*vectorSize)

	return r, nil
}

// Add inserts a vector into the registry, replacing existing vector with same ID.
func (r *Registry) Add(id uint64, vec []float32) error {
	if len(vec) != r.config.FullDim {
		return ErrDimensionMismatch
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	hybridData := QuantizeHybrid(vec, r.hybridCfg)

	if idx, exists := r.idMap[id]; exists {
		// Replace existing vector
		start := idx * r.vectorSize
		copy(r.vectors[start:start+r.vectorSize], hybridData)
		return nil
	}

	// Add new vector
	r.vectors = append(r.vectors, hybridData...)
	idx := r.totalVectors
	r.idMap[id] = idx
	r.revMap = append(r.revMap, id)
	r.totalVectors++

	return nil
}

// Get returns the compressed vector data for a given ID.
func (r *Registry) Get(id uint64) ([]byte, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	idx, exists := r.idMap[id]
	if !exists {
		return nil, false
	}

	start := idx * r.vectorSize
	result := make([]byte, r.vectorSize)
	copy(result, r.vectors[start:start+r.vectorSize])
	return result, true
}

// GetDecompressed returns the full decompressed vector for a given ID.
func (r *Registry) GetDecompressed(id uint64) ([]float32, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	idx, exists := r.idMap[id]
	if !exists {
		return nil, false
	}

	start := idx * r.vectorSize
	vecData := r.vectors[start : start+r.vectorSize]
	return DequantizeHybrid(vecData, r.config.FullDim, r.hybridCfg), true
}

// Delete removes a vector from the registry.
func (r *Registry) Delete(id uint64) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	idx, exists := r.idMap[id]
	if !exists {
		return false
	}

	lastIdx := r.totalVectors - 1
	lastID := r.revMap[lastIdx]

	if idx != lastIdx {
		// Swap with last
		srcStart := lastIdx * r.vectorSize
		dstStart := idx * r.vectorSize
		copy(r.vectors[dstStart:dstStart+r.vectorSize], r.vectors[srcStart:srcStart+r.vectorSize])

		r.revMap[idx] = lastID
		r.idMap[lastID] = idx
	}

	r.revMap = r.revMap[:lastIdx]
	delete(r.idMap, id)
	r.totalVectors--

	return true
}

// Count returns the number of vectors in the registry.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.totalVectors
}

// Has returns true if the registry contains a vector with the given ID.
func (r *Registry) Has(id uint64) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, exists := r.idMap[id]
	return exists
}

// IDs returns all vector IDs in the registry.
func (r *Registry) IDs() []uint64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]uint64, len(r.revMap))
	copy(result, r.revMap)
	return result
}

// VectorSize returns the size of each compressed vector in bytes.
func (r *Registry) VectorSize() int {
	return r.vectorSize
}

// HybridConfig returns the hybrid configuration.
func (r *Registry) HybridConfig() *HybridConfig {
	return r.hybridCfg
}

// FullDim returns the original vector dimension.
func (r *Registry) FullDim() int {
	return r.config.FullDim
}

// Config returns the registry configuration.
func (r *Registry) Config() *Config {
	return r.config
}

// Reserve pre-allocates capacity for the given number of vectors.
func (r *Registry) Reserve(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	required := n * r.vectorSize
	if required > cap(r.vectors) {
		newVec := make([]byte, len(r.vectors), required)
		copy(newVec, r.vectors)
		r.vectors = newVec
	}
}
