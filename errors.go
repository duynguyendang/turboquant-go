package turboquant

import "errors"

var (
	ErrInvalidDimension = errors.New("invalid dimension: must be positive")
	ErrInvalidBitWidth = errors.New("invalid bit width: must be 4 or 8")
	ErrInvalidBlockSize = errors.New("invalid block size: must be positive and divisible by 8")
	ErrVectorNotFound   = errors.New("vector not found")
	ErrDimensionMismatch = errors.New("dimension mismatch")
)
