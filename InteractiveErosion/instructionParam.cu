#include "instructionParam.cuh"

namespace instructionParam {

	__device__
		bool isOutside(int c, int r, int w, int h) {
		return ((c < 0) || (r<0) || (c >= w) || (r >= h));
	}

	__global__
	void d_gensum(int* dd_intParams, float** dd_terrainData) {
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) return; // Check if within image bounds
		const int cudaindex = c*DW + r;
		//sum
		float sum = 0;
		for (int i = 0; i < DLEN; ++i) {
			sum += dd_terrainData[i][cudaindex];
		}
		SUMS[cudaindex] = sum;
	}

	__global__
		void d_genslope(int* dd_intParams, float** dd_terrainData) {
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) return; // Check if within image bounds
		const int cudaindex = c*DW + r;
		//sum
		int idxs[8] = { cudaindex - DW,		cudaindex - DW + 1,		cudaindex + 1,
			cudaindex + DW + 1,							cudaindex + DW,
			cudaindex + DW - 1, cudaindex - 1,			cudaindex - DW - 1 };
		float maxdif = -10000;
		for (unsigned char i = 0; i < 8; ++i) {
			if (idxs[i] >= 0 && idxs[i] < DW*DH) {
				if (SUMS[idxs[i]] > maxdif) {
					maxdif = SUMS[idxs[i]];
				}
			}
		}
		SLOPE_SIN[cudaindex] = maxdif / sqrt(CELL_WIDTH*CELL_WIDTH+maxdif*maxdif);
	}

	int* d_intParams;
	float* d_floatParams;
	float** d_extraParams;

	clock_t sculptTimer;

	bool checkInterval() {
		if (sculptTimer == NULL) {
			sculptTimer = clock();
		}
		else {
			clock_t now = clock();
			clock_t passed = now - sculptTimer;
			float secs = (float)passed / CLOCKS_PER_SEC;
			if (secs < 0.05) {
				return false;
			}
			else {
				sculptTimer = clock();
			}
		}
		return true;
	}

	void genSum(SimMap* simMap, int* d_param) {
		const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		if (simMap == NULL) {
			printf("NULL\n\n");
		}
		const dim3 gridSize = dim3((simMap->getWidth() / BLOCK_SIZE_X) + 1, (simMap->getHeight() / BLOCK_SIZE_Y) + 1);
		d_gensum << <gridSize, blockSize >> > (d_param, simMap->getDeviceLayerDataList());
		d_genslope << <gridSize, blockSize >> > (d_param, simMap->getDeviceLayerDataList());
	}

	void passParams(SimMap* simMap, int layerIndex, float strength, float radius, float x, float y, float z,
		float sprinklerStrength, float sprinklerRadius, float evaporation) {
		cudaError_t err;
		int h_intParams[6] = { simMap->getWidth(), simMap->getHeight(), layerIndex, simMap->getLayerCount(),
							   simMap->getPoly()->getVector().size(), simMap->getSprinkler()->getVector().size()};
		float h_floatParams[8] = { strength, radius, x, y, z, sprinklerStrength, sprinklerRadius, evaporation };
		err = cudaFree(d_intParams);
		err = cudaMalloc(&d_intParams, 6 * sizeof(int));
		err = cudaMemcpy(d_intParams, h_intParams, 6 * sizeof(int), cudaMemcpyHostToDevice);
		//
		err = cudaFree(d_floatParams);
		err = cudaMalloc(&d_floatParams, 8 * sizeof(float));
		err = cudaMemcpy(d_floatParams, h_floatParams, 8 * sizeof(float), cudaMemcpyHostToDevice);
		//
	}

	int* getIntPtr() {
		return d_intParams;
	}

	float* getFloatPtr() {
		return d_floatParams;
	}

	float** getExtraPtr() {
		return d_extraParams;
	}
}