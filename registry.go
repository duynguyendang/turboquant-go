package turboquant

import (
	"iter"
	"sync"
)

// Registry stores compressed vectors in memory and provides search functionality.
type Registry struct {
	config     *Config
	hybridCfg  *HybridConfig
	vectorSize int

	vectors      []byte // contiguous memory for all compressed vectors
	idMap        map[uint64]int
	revMap       []uint64 // revMap[i] = ID of vector at slot i
	totalVectors int      // total slots used (including tombstoned)
	tombstones   []uint64 // bitset: bit i set if slot i is deleted
	tombCount    int      // number of tombstoned slots

	mu sync.RWMutex
}

// NewRegistry creates a new vector registry with the given configuration.
func NewRegistry(cfg *Config) (*Registry, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}
	if cfg.NumWorkers <= 0 {
		cfg.NumWorkers = 1
	}
	if cfg.VectorCapacity <= 0 {
		cfg.VectorCapacity = 10000
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	hybridCfg := cfg.hybridConfig()
	vectorSize := HybridVectorSize(cfg.PaddedDim(), hybridCfg)
	tombSize := (cfg.VectorCapacity + 63) / 64

	r := &Registry{
		config:     cfg,
		hybridCfg:  hybridCfg,
		vectorSize: vectorSize,
		idMap:      make(map[uint64]int, cfg.VectorCapacity),
		revMap:     make([]uint64, 0, cfg.VectorCapacity),
		tombstones: make([]uint64, tombSize),
	}

	// Pre-allocate vector storage
	r.vectors = make([]byte, 0, cfg.VectorCapacity*vectorSize)

	return r, nil
}

// tombstoneSize returns the number of uint64s needed for n slots.
func tombstoneSize(n int) int {
	return (n + 63) / 64
}

// isTombstoned checks if slot idx is tombstoned.
func (r *Registry) isTombstoned(idx int) bool {
	return (r.tombstones[idx/64] & (1 << uint(idx%64))) != 0
}

// setTombstone marks slot idx as tombstoned.
func (r *Registry) setTombstone(idx int) {
	r.tombstones[idx/64] |= 1 << uint(idx%64)
	r.tombCount++
}

// clearTombstone marks slot idx as active.
func (r *Registry) clearTombstone(idx int) {
	r.tombstones[idx/64] &^= 1 << uint(idx%64)
	r.tombCount--
}

// findTombstonedSlot returns the index of a tombstoned slot, or -1 if none.
func (r *Registry) findTombstonedSlot() int {
	for i := 0; i < len(r.tombstones); i++ {
		if r.tombstones[i] != 0 {
			for j := 0; j < 64; j++ {
				if r.tombstones[i]&(1<<uint(j)) != 0 {
					return i*64 + j
				}
			}
		}
	}
	return -1
}

// growTombstones ensures tombstones can hold at least n slots.
func (r *Registry) growTombstones(n int) {
	needed := tombstoneSize(n)
	if needed > len(r.tombstones) {
		newTs := make([]uint64, needed)
		copy(newTs, r.tombstones)
		r.tombstones = newTs
	}
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
		// Replace existing vector (un-tombstone if it was deleted)
		if r.isTombstoned(idx) {
			r.clearTombstone(idx)
		}
		start := idx * r.vectorSize
		copy(r.vectors[start:start+r.vectorSize], hybridData)
		return nil
	}

	// Try to reuse a tombstoned slot
	if idx := r.findTombstonedSlot(); idx >= 0 {
		r.clearTombstone(idx)
		start := idx * r.vectorSize
		copy(r.vectors[start:start+r.vectorSize], hybridData)
		r.idMap[id] = idx
		r.revMap[idx] = id
		return nil
	}

	// Add new vector
	idx := r.totalVectors
	r.growTombstones(idx + 1)
	r.vectors = append(r.vectors, hybridData...)
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
	if !exists || r.isTombstoned(idx) {
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
	if !exists || r.isTombstoned(idx) {
		return nil, false
	}

	start := idx * r.vectorSize
	vecData := r.vectors[start : start+r.vectorSize]
	return DequantizeHybrid(vecData, r.config.FullDim, r.hybridCfg), true
}

// Delete marks a vector as deleted (tombstone). O(1) operation.
func (r *Registry) Delete(id uint64) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	idx, exists := r.idMap[id]
	if !exists || r.isTombstoned(idx) {
		return false
	}

	r.setTombstone(idx)
	delete(r.idMap, id)
	return true
}

// Count returns the number of active (non-tombstoned) vectors.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.totalVectors - r.tombCount
}

// Has returns true if the registry contains an active vector with the given ID.
func (r *Registry) Has(id uint64) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	idx, exists := r.idMap[id]
	return exists && !r.isTombstoned(idx)
}

// IDs returns all active vector IDs in the registry.
func (r *Registry) IDs() []uint64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]uint64, 0, r.totalVectors-r.tombCount)
	for i, id := range r.revMap {
		if !r.isTombstoned(i) {
			result = append(result, id)
		}
	}
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
	r.growTombstones(n)
}

// All returns an iter.Seq2 that yields (id, compressedVector) pairs.
// Zero-copy: returns slices pointing directly into the vector slab.
// Concurrent modifications (Add/Delete) during iteration are not safe.
func (r *Registry) All() iter.Seq2[uint64, []byte] {
	r.mu.RLock()
	numVectors := r.totalVectors
	vectorSize := r.vectorSize
	revMap := r.revMap
	vectors := r.vectors
	tombstones := make([]uint64, len(r.tombstones))
	copy(tombstones, r.tombstones)
	r.mu.RUnlock()

	return func(yield func(uint64, []byte) bool) {
		for i := 0; i < numVectors; i++ {
			if tombstones[i/64]&(1<<uint(i%64)) != 0 {
				continue
			}
			start := i * vectorSize
			vec := vectors[start : start+vectorSize]
			if !yield(revMap[i], vec) {
				return
			}
		}
	}
}

// AllDecompressed returns an iter.Seq2 that yields (id, decompressedVector) pairs.
// Zero-copy for the vector slab; each decompressed vector is allocated on demand.
func (r *Registry) AllDecompressed() iter.Seq2[uint64, []float32] {
	r.mu.RLock()
	numVectors := r.totalVectors
	vectorSize := r.vectorSize
	revMap := r.revMap
	vectors := r.vectors
	tombstones := make([]uint64, len(r.tombstones))
	copy(tombstones, r.tombstones)
	dim := r.config.FullDim
	cfg := r.hybridCfg
	r.mu.RUnlock()

	return func(yield func(uint64, []float32) bool) {
		for i := 0; i < numVectors; i++ {
			if tombstones[i/64]&(1<<uint(i%64)) != 0 {
				continue
			}
			start := i * vectorSize
			vecData := vectors[start : start+vectorSize]
			vec := DequantizeHybrid(vecData, dim, cfg)
			if !yield(revMap[i], vec) {
				return
			}
		}
	}
}

// IDsIter returns an iter.Seq that yields vector IDs.
func (r *Registry) IDsIter() iter.Seq[uint64] {
	r.mu.RLock()
	numVectors := r.totalVectors
	revMap := r.revMap
	tombstones := make([]uint64, len(r.tombstones))
	copy(tombstones, r.tombstones)
	r.mu.RUnlock()

	return func(yield func(uint64) bool) {
		for i := 0; i < numVectors; i++ {
			if tombstones[i/64]&(1<<uint(i%64)) != 0 {
				continue
			}
			if !yield(revMap[i]) {
				return
			}
		}
	}
}
