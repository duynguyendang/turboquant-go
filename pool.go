package turboquant

import "sync"

var paddedPool = sync.Pool{
	New: func() any {
		return make([]float32, 4096)
	},
}

func getPaddedBuffer(size int) []float32 {
	buf := paddedPool.Get().([]float32)
	if len(buf) < size {
		buf = make([]float32, size)
	}
	return buf[:size]
}

func putPaddedBuffer(buf []float32) {
	if cap(buf) >= 1024 && cap(buf) <= 8192 {
		paddedPool.Put(buf[:cap(buf)])
	}
}
