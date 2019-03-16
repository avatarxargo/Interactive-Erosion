#pragma once
#include <windows.h>
#include "cuda.h"
#include <string>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

/**Simulation layer with allocated memory space on GPU. Destructor cleans up the allocated memory.*/
class SimLayer
{
private:
	/**pointer to the array of data on the GPU.*/
	float* d_data;
	int width, height, materialIdx;
public:
	/**Initializes only dimensions*/
	std::string name;
	SimLayer();
	SimLayer(int width, int height, std::string iname, int sMaterialIdx);
	SimLayer(int width, int height, std::string iname);
	SimLayer(const SimLayer &sl2);
	/*cudaMallocs data array, places data*/
	cudaError_t setData(float * h_initData);
	/**Generates blank data based on dimensions.*/
	cudaError_t blankData(float val);
	float* getDataPtr();
	int getMaterialIdx();
	void setMaterialIdx(int arg);
	/**Cleans up GPU mallocated areas. Call this before discarding this layer.*/
	void cleanUp();
	~SimLayer();
};