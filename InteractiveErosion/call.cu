#include "call.cuh"

#define GL_SHELL gl_Data[0]
#define GL_WALL gl_Data[1]
#define GL_MASK gl_Data[2]
#define GL_WATER gl_Data[3]
#define GL_TEXMASK gl_Data[4]
#define GL_WATERMASK gl_Data[5]
#define GL_SEDIMENT gl_Data[6]

#define DSTR dd_floatParams[0]
#define DRAD dd_floatParams[1]
#define DX dd_floatParams[2]
#define DY dd_floatParams[3]
#define DZ dd_floatParams[4]

#define DUST dd_terrainData[DLEN - 1]

#define WATER dd_terrainData[DLEN]
#define MASK dd_terrainData[DLEN+1]
#define SUMS dd_terrainData[DLEN+2]
#define WATER_LAST dd_terrainData[DLEN+3]
#define REGOLITH dd_terrainData[DLEN+4]
#define SEDIMENT dd_terrainData[DLEN+5]
#define HYDRATATION dd_terrainData[DLEN+6]
#define WATER_VERT dd_terrainData[DLEN+7]
#define WATER_HOR dd_terrainData[DLEN+8]
#define WATER_CELL_VERT dd_terrainData[DLEN+9]
#define WATER_CELL_HOR dd_terrainData[DLEN+10]
#define SLOPE_SIN dd_terrainData[DLEN+11]

#define CUR_MATERIAL d_materialData[d_materialIndex[DIDX]]
#define MATERIAL dd_materialIndex
#define MDATA dd_materialData
#define MTHERMAL [0]
#define MANGLE [1]
#define MHYDRO [2]
#define MSEDIMENT [2]

#define DVIS params[0]
#define DLEN params[1]
#define DW params[2]
#define DH params[3]

#define MATTHRESH 0
#define VSCALE 1

namespace call {

	__device__
		bool isOutside(int c, int r, int w, int h) {
		return ((c < 0) || (r < 0) || (c >= w) || (r >= h));
	}

	__device__
		bool isBorder(int c, int r, int w, int h) {
		return ((c == 0) || (r == 0) || (c == (w-1) || (r == (h-1))));
	}

	/**
	  * Sums up heights in each layer. Layers passed as an array of layer locations stored within GPU memory.
	  */
	__global__
		void d_passShell(GLfloat **gl_Data, int* params, float** dd_terrainData, float scale) {
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;
		const int glindex = 3 * (cudaindex)+1;
		int material = 0;
		float sum = 0;
		float vissum = 0;
		for (int i = 0; i < DLEN; ++i) {
			float amount = dd_terrainData[i][cudaindex];
			sum += amount;
			if (i < DVIS) {
				if (amount > MATTHRESH) {
					material = i;
				}
				vissum += amount;
			}
		}
		//shell
		GL_SHELL[glindex - 1] = c*scale;
		GL_SHELL[glindex] = (vissum / VSCALE);
		GL_SHELL[glindex + 1] = r*scale;
		//border
		if (isBorder(c, r, DW, DH)) {
			int wallindex;
			if (r == 0) {
				wallindex = c;
			} else if (c == DW - 1) {
				wallindex = DW-1+r;
			} else if (r == DH - 1) {
				wallindex = (2 * DW + DH - 3) - c;
			} else if (c == 0){
				wallindex = 2 * DW + 2 * DH - 4 - r;
			}
			int glwall = 3 * wallindex + 1;
			int offset = (2*(DW+DH)-4)*3;
			vissum = 0;
			for (int i = 0; i < 10; ++i) {
				GL_WALL[i* offset + glwall - 1] = c*scale;
				if (i <= DVIS) {
					GL_WALL[i * offset + glwall] = (vissum / VSCALE);
					vissum += dd_terrainData[i][cudaindex];
				} else {
					GL_WALL[i * offset + glwall] = 0;
				}
				GL_WALL[i * offset + glwall + 1] = r*scale;
			}
		}
		//water
		GL_WATER[glindex - 1] = c*scale;
		GL_WATER[glindex] = (sum / VSCALE) + WATER[cudaindex]/ VSCALE;
		GL_WATER[glindex + 1] = r*scale;
		//mask
		GL_MASK[cudaindex] = MASK[cudaindex];
		GL_TEXMASK[cudaindex] = material;
		GL_WATERMASK[cudaindex] = WATER[cudaindex];
		GL_SEDIMENT[cudaindex] = (SEDIMENT[cudaindex]+REGOLITH[cudaindex])*100;
	}

	__global__
		void d_test(float *gl_shellData, float* d_shellData, int len, int w, int h) {
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		const int cudaindex = c*w + r;
		const int glindex = 3 * (cudaindex)+1;
		gl_shellData[glindex] = d_shellData[cudaindex];
	}

	int* d_paramPtr;
	float** d_dataPtr;

	/**
	  * Copies heightfields from the provided map into the display buffer. Call after an update takes place and no computation is happening.
	  * Property of individual layers will
	  */
	void passShell(struct cudaGraphicsResource* shellInterlop, struct cudaGraphicsResource* wallInterlop,
		struct cudaGraphicsResource* maskInterlop, struct cudaGraphicsResource* texmaskInterlop,
		struct cudaGraphicsResource* waterInterlop, struct cudaGraphicsResource* waterMaskInterlop,
		struct cudaGraphicsResource* sedimentMaskInterlop,
		SimMap* simMap, int visibleLayers, bool showAll, float scale) {
		//glFinish();
		cudaStreamSynchronize(0);
		float *d_shell = 0;
		float *d_wall = 0;
		float *d_mask = 0;
		float *d_texmask = 0;
		float *d_water = 0;
		float *d_watermask = 0;
		float *d_sedimentmask = 0;
		cudaGraphicsMapResources(1, &shellInterlop, 0); //map
		cudaGraphicsResourceGetMappedPointer((void **)&d_shell, NULL, shellInterlop);
		cudaGraphicsMapResources(1, &wallInterlop, 0); //map
		cudaGraphicsResourceGetMappedPointer((void **)&d_wall, NULL, wallInterlop);
		cudaGraphicsMapResources(1, &maskInterlop, 0); //map
		cudaGraphicsResourceGetMappedPointer((void **)&d_mask, NULL, maskInterlop);
		cudaGraphicsMapResources(1, &texmaskInterlop, 0); //map
		cudaGraphicsResourceGetMappedPointer((void **)&d_texmask, NULL, texmaskInterlop);
		cudaGraphicsMapResources(1, &waterInterlop, 0); //map
		cudaGraphicsResourceGetMappedPointer((void **)&d_water, NULL, waterInterlop);
		cudaGraphicsMapResources(1, &waterMaskInterlop, 0); //map
		cudaGraphicsResourceGetMappedPointer((void **)&d_watermask, NULL, waterMaskInterlop);
		cudaGraphicsMapResources(1, &sedimentMaskInterlop, 0); //map
		cudaGraphicsResourceGetMappedPointer((void **)&d_sedimentmask, NULL, sedimentMaskInterlop);
		//
		if (showAll) {
			visibleLayers = simMap->getLayerCount();
		}
		//
		GLfloat* h_dataPtr[7] = {d_shell, d_wall, d_mask, d_water, d_texmask, d_watermask, d_sedimentmask};
		int h_paramPtr[4] = { visibleLayers, simMap->getLayerCount(), simMap->getWidth(), simMap->getHeight() };
		cudaFree(d_dataPtr);
		cudaMalloc(&d_dataPtr, sizeof(float*) * 7);
		cudaMemcpy(d_dataPtr, h_dataPtr, sizeof(float*) * 7, cudaMemcpyHostToDevice);
		cudaFree(d_paramPtr);
		cudaMalloc(&d_paramPtr, sizeof(int) * 4);
		cudaMemcpy(d_paramPtr, h_paramPtr, 4 * sizeof(int), cudaMemcpyHostToDevice);
		//
		const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		const dim3 gridSize = dim3((simMap->getWidth() / BLOCK_SIZE_X) + 1, (simMap->getHeight() / BLOCK_SIZE_Y) + 1);
		d_passShell << <gridSize, blockSize >> > (d_dataPtr, d_paramPtr, simMap->getDeviceLayerDataList(), scale);
		cudaGraphicsUnmapResources(1, &shellInterlop, 0); //unmap
		cudaGraphicsUnmapResources(1, &wallInterlop, 0); //unmap
		cudaGraphicsUnmapResources(1, &maskInterlop, 0); //unmap
		cudaGraphicsUnmapResources(1, &texmaskInterlop, 0); //unmap
		cudaGraphicsUnmapResources(1, &waterInterlop, 0); //unmap
		cudaGraphicsUnmapResources(1, &waterMaskInterlop, 0); //unmap
		cudaGraphicsUnmapResources(1, &sedimentMaskInterlop, 0); //unmap
		cudaStreamSynchronize(0);
		cudaError_t err = cudaGetLastError();
	}
}