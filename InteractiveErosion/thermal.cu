#include "thermal.cuh"

#define TOREMOVE target[8]

#define PI 3.14159
#define DEG *(180/PI)

namespace thermal {
	__device__
		unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

	__device__
		float fun(float n) { return n+n; }

	__device__
		bool isOutside(int c, int r, int w, int h) {
		return ((c < 0) || (r<0) || (c >= w) || (r >= h));
	}

	__device__
		int idx(int row, int col, int w) {
		return row*w + col;
	}

	/**thermal erodes a single layer of the terrain. Returns true if the material in the layer has been sufficient and no further layer shall be processed.*/
	__device__
	bool d_evalLayer(float* myh, float* other, bool* other_valid, float layer_volume, float thermal_rate, float angle,
					 bool * out_valid, float * out_sum, float * out_maxdif, float * out_target) {
		bool retval = true;
		*out_sum = 0;
		*out_maxdif = 0;
		//evaluate neighbours
		for (unsigned char i = 0; i < 8; ++i) {
			float diff = *myh - other[i];
			if (other_valid[i] && diff>0 && atan2f(diff, CELL_WIDTH) > (angle * PI) / 180) {
				*out_sum += diff;
				out_valid[i] = true;
				if (diff > *out_maxdif) {
					*out_maxdif = diff;
				}
			}
			else {
				out_valid[i] = false;
			}
		}
		//distribute material
		float amount = thermal_rate * *out_maxdif / 2;
		//check if enough material is present for our cell.
		if (amount > layer_volume) {
			amount = layer_volume;
			retval = false;
		}
		//stop if the 
		if (amount == 0) {
			return;
		}
		//disperse the available material.
		for (unsigned char i = 0; i < 8; ++i) {
			if (out_valid[i]) {
				out_target[i] += amount * ((*myh - other[i]) / *out_sum);
			}
		}
		out_target[8] += amount;
		//reduce the height for the next step.
		*myh -= amount;
		return retval;
	}

	__device__
		float d_gaussFalloff(float x, float y, float cx, float cy, float radius) {
		return expf(-((((x - cx)*(x - cx) + (y - cy)*(y - cy)) * 4) / (radius*radius)));
	}

	__global__
		void d_thermalKernel1a(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float* dd_working, int* dd_materialIndex, float** dd_materialData) {
		//extraFields
		//0 - sums of all layer heights
		//1 - transport array
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) return; // Check if within image bounds
		const int cudaindex = c*DW + r;
		//update

		// thermal erosion
		/*  x(row) -->
		y(col)	[8,1,2]
		|		[7,H,3]
		V		[6,5,4]
		*/
		//neighbours
		float myh = SUMS[cudaindex];
		int idxs[8] = { cudaindex - DW,		cudaindex - DW + 1,		cudaindex + 1,
						cudaindex + DW + 1,							cudaindex + DW,
						cudaindex + DW - 1, cudaindex - 1,			cudaindex - DW - 1 };
		bool other_valid[8];
		bool validtmp[8] = { true,true, true, true, true, true, true, true };
		float target[9] = { 0,0,0,0,0,0,0,0,0 };
		float other[8] = {0,0,0,0,0,0,0,0};
		for (unsigned char i = 0; i < 8; ++i) {
			if (idxs[i] >= 0 && idxs[i] < DW*DH && (r > 0 || i < 5) && (r < DW-1 || (i!=1 && i !=2 && i!=3)))	{
				other[i] = SUMS[idxs[i]];
				other_valid[i] = true;
			}
			else {
				other_valid[i] = false;
			}
		}
		float sum = 0, maxdif = 0;
		//Start eroding each layer from the top most:
		for (int i = DLEN-1; i >= 0; --i) {
			//If the layer is empty. Skip
			if (dd_terrainData[i][cudaindex] == 0) { continue; }
			//If the layer material was sufficient, no further erosion takes place.
			if (d_evalLayer(&myh, other, other_valid, dd_terrainData[i][cudaindex], MDATA[MATERIAL[i]]MTHERMAL, MDATA[MATERIAL[i]]MANGLE,
				validtmp, &sum, &maxdif, target)) {
				break;
			}
		}		
		dd_working[cudaindex * 9] = TOREMOVE;
		//Better Pass
		for (unsigned char i = 0; i < 8; ++i) {
			if(other_valid[i])
				dd_working[idxs[i] * 9 + 1 + i] = target[i];
		}
		//
		// -----------------------
		//CONTINUE AFTER CPU SYNCH...
	}

	__global__
		void d_thermalKernel1b(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float* dd_working, int* d_materialIndex, float** d_materialData) {
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) return; // Check if within image bounds
		const int cudaindex = c*DW + r;
		int tidx = cudaindex * 9;
		//update
		//...CONTINUING AFTER CPU SYNCH
		// -----------------------
		//topidx
		int topidx = 0;
		for (unsigned char i = 0; i < DLEN; ++i) {
			if (dd_terrainData[i][cudaindex] > 0) {
				topidx = i;
			}
		}
		//remove material first
		float sum = dd_working[tidx];
		for (unsigned char i = DLEN - 1; i >= 0 ; --i) {
			if (dd_terrainData[i][cudaindex] >= sum) {
				dd_terrainData[i][cudaindex] -= sum;
				break;
			} else {
				sum -= dd_terrainData[i][cudaindex];
				dd_terrainData[i][cudaindex] = 0;
			}
		}
		//Add up the deltas.
		sum = 0;
		for (unsigned int i = 1; i < 9; ++i) {
			sum += dd_working[tidx + i];
		}
		DUST[cudaindex] += sum;
	}

	instructionParam::myCudaParam_t params;
	std::mutex m;
	std::condition_variable cv;
	unsigned int requested = 0;
	bool active = true;
	volatile bool activityRequest = false;
	
	void activity() {
		//lock simMap
		DWORD dwWaitResult;
		dwWaitResult = WaitForSingleObject(
			params.handle,    // handle to mutex
			INFINITE);  // no time-out interval
						//
						//create working arrays
		instructionParam::passParams(params.simMap, params.idx, params.strength, params.radius, params.x, params.y, params.z, params.sprinklerStrength, params.sprinklerRadius, params.evaporation);
		instructionParam::genSum(params.simMap, instructionParam::getIntPtr());
		float * d_working;
		int worksize = 9 + 9 * params.simMap->getWidth() * params.simMap->getHeight();
		float * h_working = new float[worksize];
		for (int i = 0; i < worksize; ++i) {
			h_working[i] = 0;
		}
		cudaError_t err = cudaMalloc(&d_working, worksize * sizeof(float));
		err = cudaMemcpy(d_working, h_working, worksize * sizeof(float), cudaMemcpyHostToDevice);
		//
		const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		if (params.simMap == NULL) {
			printf("NULL\n\n");
		}
		const dim3 gridSize = dim3((params.simMap->getWidth() / BLOCK_SIZE_X) + 1, (params.simMap->getHeight() / BLOCK_SIZE_Y) + 1);
		d_thermalKernel1a << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		//CPU synch
		d_thermalKernel1b << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		//callback after completion
		if (params.callback != NULL) {
			params.callback();
		}
		//free
		err = cudaFree(d_working);
		free(h_working);
		//unlock simMap
		ReleaseMutex(params.handle);
		err = cudaGetLastError();
		if (err != 0) {
			printf("XXX Thermal CUDA encountered an error number %d! \n", err);
		}
	}

	void worker_thread()
	{
		while (true) {
			// Wait until main() sends data
			std::unique_lock<std::mutex> lk(m);
			cv.wait(lk, [] {return activityRequest; });
			if (!active) { lk.unlock(); return; }

			// after the wait, we own the lock.
			activity();

			// Manual unlocking is done before notifying, to avoid waking up
			// the waiting thread only to block again (see notify_one for details)
			lk.unlock();
			activityRequest = false;
		}
	}
	
	void killThread() {
		printf("Terminating Thermal Thread\n");
		active = false;
		cv.notify_one();
	}

	void initThread() {
		printf("Initing Thermal Thread\n");
		std::thread worker(worker_thread);
		worker.detach();
	}

	void erodeThermal(float x, float y, float z, int idx, float toolStregth, float toolRadius, float dir, SimMap* simMap, HANDLE handle, void(*callback)(), float sprinklerStrength, float  sprinklerRadius, float evaporation) {
		params = { x,y,z,toolRadius, toolStregth*dir, idx, simMap,handle,callback, sprinklerStrength, sprinklerRadius, evaporation };
		//ping thread
		activityRequest = true;
		cv.notify_one();
	}
}