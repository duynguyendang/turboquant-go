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

// putUint64 writes a uint64 in little-endian format.
func putUint64(b []byte, v uint64) {
	binary.LittleEndian.PutUint64(b, v)
}

// getUint64 reads a uint64 from little-endian format.
func getUint64(b []byte) uint64 {
	return binary.LittleEndian.Uint64(b)
}

// putUint16 writes a uint16 in little-endian format.
func putUint16(b []byte, v uint16) {
	binary.LittleEndian.PutUint16(b, v)
}

// getUint16 reads a uint16 from little-endian format.
func getUint16(b []byte) uint16 {
	return binary.LittleEndian.Uint16(b)
}
