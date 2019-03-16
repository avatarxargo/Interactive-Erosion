#include "sculpt.cuh"

namespace sculpt {

	__device__
		float d_gaussFalloff(float x, float y, float cx, float cy, float radius) {
		float val = expf(-((((x - cx)*(x - cx) + (y - cy)*(y - cy)) * 4) / (radius*radius)));
		return val < 0.01 ? 0 : val;
	}

	__device__
		bool isOutside(int c, int r, int w, int h) {
		return ((c < 0) || (r<0) || (c >= w) || (r >= h));
	}

	__global__
		void d_sculptHeight(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, int* d_materialIndex, float** d_materialData) {
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;
		//update
		float val = dd_terrainData[DIDX][cudaindex] + MASK[cudaindex] * DSTR * d_gaussFalloff(c, r, DX, DZ, DRAD) * 0.01;
		if (val < 0) {
			val = 0;
		}
		dd_terrainData[DIDX][cudaindex] = val;
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
		instructionParam::passParams(params.simMap, params.idx, params.strength, params.radius, params.x, params.y, params.z, params.sprinklerStrength, params.sprinklerRadius, params.evaporation);
		const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		if (params.simMap == NULL) {
			printf("NULL\n\n");
		}
		const dim3 gridSize = dim3((params.simMap->getWidth() / BLOCK_SIZE_X) + 1, (params.simMap->getHeight() / BLOCK_SIZE_Y) + 1);
		d_sculptHeight << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		if (params.callback != NULL) {
			params.callback();
		}
		cudaError_t err = cudaGetLastError();
		if (err != 0) {
			printf("XXX Sculpt CUDA encountered an error number %d! \n", err);
		}
		//unlock simMap
		ReleaseMutex(params.handle);
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
		printf("Terminating Sculpt Thread\n");
		active = false;
		cv.notify_one();
	}

	void initThread() {
		printf("Initing Sculpt Thread\n");
		std::thread worker(worker_thread);
		worker.detach();
	}

	void sculptHeight(float x, float y, float z, int idx, float toolStregth, float toolRadius, float dir, SimMap* simMap, HANDLE handle, void(*callback)(), float sprinklerStrength, float  sprinklerRadius, float evaporation) {
		params = { x,y,z,toolRadius, toolStregth*dir, idx, simMap,handle,callback, sprinklerStrength, sprinklerRadius, evaporation };
		//ping thread
		activityRequest = true;
		cv.notify_one();
	}
}