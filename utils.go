package turboquant

import (
	"encoding/binary"
	"math"
)

// putFloat32 writes a float32 in little-endian format.
func putFloat32(b []byte, v float32) {
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
}

// getFloat32 reads a float32 from little-endian format.
func getFloat32(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}
