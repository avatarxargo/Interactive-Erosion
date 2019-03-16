#include "hydro.cuh"

namespace hydro {


	__device__
		float d_gaussFalloff(float x, float y, float cx, float cy, float radius) {
		float val = expf(-((((x - cx)*(x - cx) + (y - cy)*(y - cy)) * 4) / (radius*radius)));
		return val < 0.01 ? 0 : val;
	}

	__device__
		bool isOutside(int c, int r, int w, int h) {
		return ((c < 0) || (r<0) || (c >= w) || (r >= h));
	}

	__device__
		bool isBorder(int c, int r, int w, int h) {
		return ((c == 0) || (r == 0) || (c == (w - 1) || (r == (h - 1))));
	}

	__global__
		void d_erodeWaterHydraulic(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* dd_materialIndex, float** dd_materialData) {
		// Prerequisites: call genSlopes() and conduct a water flow update to have the latest WATER_LAST and SLOPE_SIN before execution.
		// dissolves material into particles from the top.

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;

		//hydraulic erosion
		float dissolutionCapacity = 0.001;
		if (dissolutionCapacity > WATER[cudaindex]) {
			dissolutionCapacity = WATER[cudaindex];
		}

		if (REGOLITH[cudaindex] > dissolutionCapacity) {
			DUST[cudaindex] += REGOLITH[cudaindex] - dissolutionCapacity;
			REGOLITH[cudaindex] = dissolutionCapacity;
		}
		else {
			float remainingHydro = dissolutionCapacity - REGOLITH[cudaindex];
			for (int i = DLEN - 1; i >= 0; --i) {
				const float requestHydro = /*MDATA[MATERIAL[i]]MHYDRO */ remainingHydro;
				if (requestHydro > dd_terrainData[i][cudaindex]) {
					dd_terrainData[i][cudaindex] = 0;
					remainingHydro -= requestHydro;
				}
				else {
					dd_terrainData[i][cudaindex] -= requestHydro;
					remainingHydro -= requestHydro;
					break;
				}
			}
			REGOLITH[cudaindex] += dissolutionCapacity - remainingHydro - REGOLITH[cudaindex];
		}
	}

	__global__
		void d_erodeWaterKinetic(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* dd_materialIndex, float** dd_materialData) {
		// Prerequisites: call genSlopes() and conduct a water flow update to have the latest WATER_LAST and SLOPE_SIN before execution.
		// dissolves material into particles from the top.

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;

		//sediment transport capacity
		//S = v * C * sin(alpha)
		const float velocity = sqrt(WATER_CELL_VERT[cudaindex] * WATER_CELL_VERT[cudaindex] + WATER_CELL_HOR[cudaindex] * WATER_CELL_HOR[cudaindex]);
		//set minimal transport capacity with respect to slope to avoid no effect on flat surfaces.
		float transportCapacity = 0.1;
		if (SLOPE_SIN[cudaindex] < 0.1) {
			transportCapacity *= velocity * 0.1;
		}
		else {
			transportCapacity *= velocity * SLOPE_SIN[cudaindex];
		}
		const float DISSOLVE = 1;
		const float SETTLE = 1;
		//compate material levels to capacity
		if (SEDIMENT[cudaindex] > transportCapacity) {
			//deposit
			float delta = SETTLE*(SEDIMENT[cudaindex] - transportCapacity);
			DUST[cudaindex] += delta;
			SEDIMENT[cudaindex] -= delta;
		}
		else {
			//erode
			float delta = DISSOLVE*(transportCapacity - SEDIMENT[cudaindex]);
			//start removing material from the terrain.
			float remaining = delta - SEDIMENT[cudaindex];
			for (int i = DLEN - 1; i >= 0; --i) {
				 const float request = MDATA[MATERIAL[i]]MHYDRO * remaining;
				 if(request > dd_terrainData[i][cudaindex]) {
					 dd_terrainData[i][cudaindex] = 0;
					 remaining -= request;
				 }
				 else {
					 dd_terrainData[i][cudaindex] -= request;
					 remaining -= request;
					 break;
				 }
			}
			//
			SEDIMENT[cudaindex] += delta-remaining;
		}
	}

	__global__
		void d_erodeWaterSemiLag1(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* dd_materialIndex, float** dd_materialData) {

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;

		//move material
		const float scale = 20;
		const float oldx = c - (WATER_CELL_HOR[cudaindex] * scale);
		const float oldy = r -(WATER_CELL_VERT[cudaindex] * scale);
		const int foldx = floorf(oldx);
		const int foldy = floorf(oldy);

		//weight with which to pick each 
		float wx = oldx - foldx;
		float wy = oldy - foldy;
		float old = (oldy)*DW + (oldx);
		const int idx[4] = { (foldy)*DW + (foldx), (foldy)*DW + (foldx+1), (foldy+1)*DW + (foldx), (foldy+1)*DW + (foldx+1) };
		const bool valid[4] = { idx[0]>0 && idx[0]<DW*DH, idx[1]>0 && idx[1]<DW*DH, idx[2]>0 && idx[2]<DW*DH, idx[3]>0 && idx[3]<DW*DH };
		//
		// [foldx, foldy]	  wx >	 [foldx+1, foldy]
		//		 wy V
		// [foldx, foldy+1]			 [foldx+1, foldy+1]
		//
		float amountSediment = 0;
		float amountRegolith = 0;
		float div;
		if (valid[0]) {	amountSediment += SEDIMENT[idx[0]] * (1 - wy)*(1 - wx);
						amountRegolith += REGOLITH[idx[0]] * (1 - wy)*(1 - wx);	}
		if (valid[1]) { amountSediment += SEDIMENT[idx[1]] * (1 - wy)*(wx);	
						amountRegolith += REGOLITH[idx[1]] * (1 - wy)*(wx); }
		if (valid[2]) { amountSediment += SEDIMENT[idx[2]] * (wy)*(1 - wx);
						amountRegolith += REGOLITH[idx[2]] * (wy)*(1 - wx); }
		if (valid[3]) { amountSediment += SEDIMENT[idx[3]] * (wy)*(wx);
						amountRegolith += REGOLITH[idx[3]] * (wy)*(wx); }
		for (unsigned int i = 0; i < 4; ++i) {
			if (valid[i]) {
				++div;
			}
		}
		//compensate for missing cells
		if (div > 0) {
			amountSediment *= 4 / div;
			d_extra[cudaindex] = amountSediment;
			d_extra[DW*DH+cudaindex] = amountRegolith;
		}
		//
		// Continue after CPU synch
		// ...
		//
	}

	__global__
		void d_erodeWaterSemiLag2(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* dd_materialIndex, float** dd_materialData) {
		// just adds up the values computed temporarily

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;

		SEDIMENT[cudaindex] = d_extra[cudaindex];
		REGOLITH[cudaindex] = d_extra[DW*DH + cudaindex];
	}

	__global__
		void d_simWaterA(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* d_materialIndex, float** d_materialData) {
		// 1 / 4 - compute pressure differences for each CELL. OUT: Pass accelerations to pipes

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;
		const int offset = (DW*DH);

		//add amount from sprinklers
		float val = 0;
		for (unsigned int i = 0; i < DPOLY_SPR_LEN; ++i) {
			val += DSPRINKLER_STR * d_gaussFalloff(c, r, POLY_SPR[i*3] * 2, POLY_SPR[i*3+2] * 2, DSPRINKLER_RADIUS);
		}
		WATER[cudaindex] += val;

		//pipe indexes: WEST, NORTH, EAST, SOUTH
		const int idx[4] = { cudaindex - 1, cudaindex - DW, cudaindex+1, cudaindex + DW };
		bool valid[4] = { idx[0] >= 0, idx[1] >= 0, idx[2] < DW*DH, idx[3] < DW*DH };
		const int dir[4] = { -1, -1, 1, 1 };
		const int pipeidx[4] = { cudaindex - 1, offset + cudaindex - DW, cudaindex, offset + cudaindex };
		//bool pipevalid[4] = { pipeidx[0] >= 0 && pipeidx[0]%DW!=DW-1, pipeidx[1] - offset >= 0, pipeidx[2] < DW*DH, pipeidx[3] - offset < DW*(DH-1) };
		float dif[4] = { 0,0,0,0 };

		//compute
		float myh = SUMS[cudaindex] + WATER[cudaindex] + SEDIMENT[cudaindex] + REGOLITH[cudaindex];
		for (unsigned char i = 0; i < 4; ++i) {
			if (valid[i]) {
				dif[i] = myh - SUMS[idx[i]] - WATER[idx[i]] - SEDIMENT[idx[i]] - REGOLITH[idx[i]];
				if (dif[i] < 0) { //do not write into higher slots. (only write from above)
					dif[i] = 0;
					valid[i] = false;
				}
			}
		}

		//store accelerations
		for (unsigned char i = 0; i < 4; ++i) {
			if (valid[i]) {
				d_extra[pipeidx[i]] = (float)(dir[i] * dif[i]);
			}
		}
	}

	__global__
		void d_simWaterB(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* d_materialIndex, float** d_materialData) {
		// 2 / 4 - add accelerations to pipe values

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;
		const int offset = DW*DH;

		//add accelerations
		WATER_HOR[cudaindex] += d_extra[cudaindex];
		WATER_VERT[cudaindex] += d_extra[offset + cudaindex];
		//nullify if source has less water.
		if (WATER_HOR[cudaindex] > 0) {
			if (WATER[cudaindex] < WATER_HOR[cudaindex]) { WATER_HOR[cudaindex] = WATER[cudaindex]; }
		}
		else {
			if (WATER[cudaindex+1] < -WATER_HOR[cudaindex]) { WATER_HOR[cudaindex] = -WATER[cudaindex + 1]; }
		}
		//
		if (WATER_VERT[cudaindex] > 0) {
			if (WATER[cudaindex] < WATER_VERT[cudaindex]) { WATER_VERT[cudaindex] = WATER[cudaindex]; }
		}
		else {
			if (WATER[cudaindex + DW] < -WATER_VERT[cudaindex]) { WATER_VERT[cudaindex] = -WATER[cudaindex + DW]; }
		}
	}

	__global__
		void d_simWaterC(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* d_materialIndex, float** d_materialData) {
		// 3 / 4 - compute transported amounts of water in each cell based on speed in the pipes.

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;
		const int extraidx = 5 * cudaindex;
		const int offset = (DW*DH);

		//transfer speeds
		const int pipeidx[4] = { cudaindex - 1, cudaindex - DW, cudaindex, cudaindex };
		const char pipedir[4] = { -1, -1, 1, 1 };
		const bool pipevalid[4] = { pipeidx[0] >= 0 && pipeidx[0] % DW != DW - 1, pipeidx[1] >= 0, pipeidx[2] < DW*DH, pipeidx[3] < DW*(DH - 1) };
		const int idx[4] = { cudaindex - 1, cudaindex - DW, cudaindex + 1, cudaindex + DW };
		//const bool valid[4] = { idx[0] >= 0, idx[1] >= 0, idx[2] < DW*DH, idx[3] < DW*DH };
		float dif[4] = { 0,0,0,0 };
		//see demands
		float sum = 0;
		if (pipevalid[0]) {
			dif[0] = -WATER_HOR[pipeidx[0]];
		}
		if (pipevalid[1]) {
			dif[1] = -WATER_VERT[pipeidx[1]];
		}
		if (pipevalid[2]) {
			dif[2] = WATER_HOR[pipeidx[2]];
		}
		if (pipevalid[3]) {
			dif[3] = WATER_VERT[pipeidx[3]];
		}
		for (unsigned char i = 0; i < 4; ++i) {
			if (dif[i] < 0) {
				dif[i] = 0;
			}
			else {
				sum += dif[i];
			}
		}
		if (sum == 0) { return; }
		float amount = sum;
		if (amount > WATER[cudaindex]) {
			amount = WATER[cudaindex];
		}
		amount /= 2;
		for (unsigned char i = 0; i < 4; ++i) {
			if (pipevalid[i]) {
				dif[i] = amount * (dif[i] / sum);
				d_extra[idx[i] * 5 + 1 + i] = dif[i];
			}
		}
		d_extra[extraidx] = amount;
		//calculate overal flows for use in kinetic hydro-erosion
		WATER_CELL_VERT[cudaindex] = dif[3] - dif[1];
		WATER_CELL_HOR[cudaindex] = dif[2] - dif[0];
	}

	__global__
		void d_simWaterD(int* dd_intParams, float* dd_floatParams, float** dd_terrainData, float * d_extra, int* d_materialIndex, float** d_materialData) {
		// 4 / 4 - Add up change in water volumes.

		//indexing
		const int c = blockIdx.x*blockDim.x + threadIdx.x;
		const int r = blockIdx.y*blockDim.y + threadIdx.y;
		if (isOutside(c, r, DW, DH)) { return; }
		const int cudaindex = r*DW + c;
		const int extraidx = 5 * cudaindex;
		//sum
		float sum = -d_extra[extraidx];
		for (unsigned char i = 0; i < 4; ++i) {
			sum += d_extra[extraidx + 1 + i];
		}
		if (isBorder(c, r, DW, DH)) {
			WATER[cudaindex] = 0;
			SEDIMENT[cudaindex] = 0;
			REGOLITH[cudaindex] = 0;
		}
		else {
			float evaporate = DEVAPORATION/1000;
			if (evaporate < 0) { evaporate = 0; }
			WATER[cudaindex] += sum - evaporate;
			if (WATER[cudaindex] < 0) { WATER[cudaindex] = 0; }
			SEDIMENT[cudaindex] -= evaporate * 2;
			if (SEDIMENT[cudaindex] < 0) { SEDIMENT[cudaindex] = 0; }
		}
		WATER_LAST[cudaindex] = (WATER[cudaindex] + WATER_LAST[cudaindex]) / 2;
	}

	instructionParam::myCudaParam_t params;
	std::mutex m;
	std::condition_variable cv;
	unsigned int requested = 0;
	bool active = true;
	bool a_kinetic = true;
	bool a_hydraulic = true;
	volatile bool activityRequest = false;

	void activity() {
		//lock simMap
		DWORD dwWaitResult;
		dwWaitResult = WaitForSingleObject(
			params.handle,    // handle to mutex
			100);  // no time-out interval
		if (dwWaitResult != WAIT_OBJECT_0) {
			return;
		}
		instructionParam::passParams(params.simMap, params.idx, params.strength, params.radius, params.x, params.y, params.z, params.sprinklerStrength, params.sprinklerRadius, params.evaporation);
		cudaError_t err = cudaGetLastError();
		if (err != 0) {
			printf("XXX PARAM Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		instructionParam::genSum(params.simMap, instructionParam::getIntPtr());
		err = cudaGetLastError();
		if (err != 0) {
			printf("XXX SUM Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		cudaDeviceSynchronize();
		float * d_working;
		int worksize = 9 * params.simMap->getWidth() * params.simMap->getHeight();
		float * h_working = new float[worksize];
		for (int i = 0; i < worksize; ++i) {
			h_working[i] = 0;
		}
		err = cudaMalloc(&d_working, worksize * sizeof(float));
		err = cudaMemcpy(d_working, h_working, worksize * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0) {
			printf("XXX COPY 1 Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		//
		const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		if (params.simMap == NULL) {
			printf("NULL\n\n");
		}
		const dim3 gridSize = dim3((params.simMap->getWidth() / BLOCK_SIZE_X) + 1, (params.simMap->getHeight() / BLOCK_SIZE_Y) + 1);
		d_simWaterA << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		if (err != 0) {
			printf("XXX WATERA Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		//CPU synchro
		d_simWaterB << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		if (err != 0) {
			printf("XXX WATERB Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		//cleanse memory, we are repurposing it from now on.
		cudaDeviceSynchronize();
		err = cudaMemcpy(d_working, h_working, worksize * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0) {
			printf("XXX COPY 2 Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		//CPU synchro
		d_simWaterC << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		//CPU synchro
		d_simWaterD << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		//
		//Now that water has been updated compute erosion
		//
		//cleanse memory, we are repurposing it from now on.
		cudaDeviceSynchronize();
		err = cudaMemcpy(d_working, h_working, worksize * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0) {
			printf("XXX COPY 3 Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		//
		if (a_kinetic) {
			d_erodeWaterKinetic << <gridSize, blockSize >> > (
				instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
				params.simMap->getDeviceLayerDataList(),
				d_working,
				params.simMap->getDeviceLayerMaterialIndexList(),
				params.simMap->getDeviceMaterialDataList());
		}
		//
		if (a_hydraulic) {
			d_erodeWaterHydraulic << <gridSize, blockSize >> > (
				instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
				params.simMap->getDeviceLayerDataList(),
				d_working,
				params.simMap->getDeviceLayerMaterialIndexList(),
				params.simMap->getDeviceMaterialDataList());
		}
		//CPU synchro
		d_erodeWaterSemiLag1 << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		//CPU synchro
		d_erodeWaterSemiLag2 << <gridSize, blockSize >> > (
			instructionParam::getIntPtr(), instructionParam::getFloatPtr(),
			params.simMap->getDeviceLayerDataList(),
			d_working,
			params.simMap->getDeviceLayerMaterialIndexList(),
			params.simMap->getDeviceMaterialDataList());
		//
		if (params.callback != NULL) {
			params.callback();
		}
		//free
		err = cudaFree(d_working);
		if (err != 0) {
			printf("XXX FREE Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
		}
		free(h_working);
		cudaStreamSynchronize(0);
		//unlock simMap
		ReleaseMutex(params.handle);
		err = cudaGetLastError();
		if (err != 0) {
			printf("XXX Hydraulic CUDA encountered an error number %d! \n", err);
			return;
			exit(-1);
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
			//std::cout << "Hydro worker thread is active.\n";
			activity();

			// Manual unlocking is done before notifying, to avoid waking up
			// the waiting thread only to block again (see notify_one for details)
			lk.unlock();
			activityRequest = false;
		}
	}

	void killThread() {
		printf("Terminating Hydro Thread\n");
		active = false;
		cv.notify_one();
	}

	void initThread() {
		printf("Initing Hydro Thread\n");
		std::thread worker(worker_thread);
		worker.detach();
	}

	void simWater(float x, float y, float z, bool kinetic, bool hydraulic, int idx, float toolStregth, float toolRadius, float dir, SimMap* simMap, HANDLE handle, void(*callback)(), float sprinklerStrength, float  sprinklerRadius, float evaporation) {
		params = { x,y,z,toolRadius, toolStregth*dir, idx, simMap,handle,callback, sprinklerStrength, sprinklerRadius, evaporation };
		a_kinetic = kinetic;
		a_hydraulic = hydraulic;
		//ping thread
		activityRequest = true;
		cv.notify_one();
	}
}