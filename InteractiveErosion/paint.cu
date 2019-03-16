#include "paint.cuh"

#define DW dd_intParams[0]
#define DH dd_intParams[1]
#define DIDX dd_intParams[2]
#define DLEN dd_intParams[3]

#define DSTR dd_floatParams[0]
#define DRAD dd_floatParams[1]
#define DX dd_floatParams[2]
#define DY dd_floatParams[3]
#define DZ dd_floatParams[4]

#define WATER dd_terrainData[DLEN]
#define MASK dd_terrainData[DLEN+1]
#define CUR_MATERIAL d_materialData[d_materialIndex[DIDX]]

namespace paint {

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
		void d_paintMask(bool paintType, int* dd_intParams, float* dd_floatParams, float** dd_terrainData, int* d_materialIndex, float** d_materialData) {
		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;
		//update
		float val =  DSTR/1000 * d_gaussFalloff(c, r, DX, DZ, DRAD);
		if (paintType) {
			val += MASK[cudaindex];
			if (val > 1) { val = 1; }
			if (val < 0) { val = 0; }
			MASK[cudaindex] = val;
		}
		else {
			val += WATER[cudaindex];
			if (val < 0) { val = 0; }
			WATER[cudaindex] = val;
		}
	}


	instructionParam::myCudaParam_t params;
	bool paintOpt = true;
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
		d_paintMask << <gridSize, blockSize >> > (
			paintOpt,
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			//instructionParam::getExtraPtr(),
			params.simMap->getDeviceLayerDataList(),
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		if (params.callback != NULL) {
			params.callback();
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
		printf("Terminating Paint Thread\n");
		active = false;
		cv.notify_one();
	}

	void initThread() {
		printf("Initing Thermal Thread\n");
		std::thread worker(worker_thread);
		worker.detach();
	}

	void paintMask(bool maskOrWater, float x, float y, float z, int idx, float toolStregth, float toolRadius, float dir, SimMap* simMap, HANDLE handle, void(*callback)(), float sprinklerStrength, float  sprinklerRadius, float evaporation) {
		//simMap->passMaterialListPointers();
		paintOpt = maskOrWater;
		params = { x,y,z,toolRadius, toolStregth*dir, idx, simMap,handle,callback, sprinklerStrength, sprinklerRadius, evaporation };
		//ping thread
		activityRequest = true;
		cv.notify_one();
	}
}