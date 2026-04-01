//go:build amd64 && !noasm

package turboquant

func dotProductHybridBatch(query []byte, vectors []byte, numVectors int, vectorSize int, dim int, cfg *HybridConfig, scores []float32) {
	if cfg == nil {
		cfg = DefaultHybridConfig()
	}

	paddedDim := nextPow2(dim)
	blockSize := cfg.BlockSize
	numBlocks := (paddedDim + blockSize - 1) / blockSize

	switch cfg.BitWidth {
	case BitWidth8:
		dotProductHybridBatch8(query, vectors, numVectors, vectorSize, numBlocks, blockSize, cfg, scores)
	case BitWidth4:
		dotProductHybridBatch4(query, vectors, numVectors, vectorSize, numBlocks, blockSize, cfg, scores)
	}
}

func dotProductHybridBatch8(query []byte, vectors []byte, numVectors int, vectorSize int, numBlocks int, blockSize int, cfg *HybridConfig, scores []float32) {
	paddedDim := numBlocks * blockSize

	for i := 0; i < numVectors; i++ {
		vecStart := i * vectorSize
		vecData := vectors[vecStart : vecStart+vectorSize]

		var totalSum float32
		offsetA := 0
		offsetB := 0

		for block := 0; block < numBlocks; block++ {
			blockLen := blockSize
			if (block+1)*blockSize > paddedDim {
				blockLen = paddedDim - block*blockSize
			}

			scaleA := getFloat32(query[offsetA:])
			zeroA := getFloat32(query[offsetA+4:])
			scaleB := getFloat32(vecData[offsetB:])
			zeroB := getFloat32(vecData[offsetB+4:])

			offsetA += 8
			offsetB += 8

			sumQQ, sumQA, sumQB := dotProdBlock8(query[offsetA:offsetA+blockLen], vecData[offsetB:offsetB+blockLen])
			offsetA += blockSize
			offsetB += blockSize

			totalSum += scaleA*scaleB*float32(sumQQ) +
				scaleA*zeroB*float32(sumQA) +
				scaleB*zeroA*float32(sumQB) +
				float32(blockLen)*zeroA*zeroB
		}

		scores[i] = totalSum
	}
}

func dotProductHybridBatch4(query []byte, vectors []byte, numVectors int, vectorSize int, numBlocks int, blockSize int, cfg *HybridConfig, scores []float32) {
	paddedDim := numBlocks * blockSize

	for i := 0; i < numVectors; i++ {
		vecStart := i * vectorSize
		vecData := vectors[vecStart : vecStart+vectorSize]

		var totalSum float32
		offsetA := 0
		offsetB := 0

		for block := 0; block < numBlocks; block++ {
			blockLen := blockSize
			if (block+1)*blockSize > paddedDim {
				blockLen = paddedDim - block*blockSize
			}

			scaleA := getFloat32(query[offsetA:])
			zeroA := getFloat32(query[offsetA+4:])
			scaleB := getFloat32(vecData[offsetB:])
			zeroB := getFloat32(vecData[offsetB+4:])

			offsetA += 8
			offsetB += 8

			sumQQ, sumQA, sumQB := dotProdBlock4(query[offsetA:offsetA+blockLen/2], vecData[offsetB:offsetB+blockLen/2], blockLen)
			offsetA += blockSize / 2
			offsetB += blockSize / 2

			totalSum += scaleA*scaleB*float32(sumQQ) +
				scaleA*zeroB*float32(sumQA) +
				scaleB*zeroA*float32(sumQB) +
				float32(blockLen)*zeroA*zeroB
		}

		scores[i] = totalSum
	}
}

func dimFromBlocks(numBlocks, blockSize int) int {
	return numBlocks * blockSize
}
