#pragma once
#include <windows.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "SimMap.cuh"
#include "globals.h"
#include <ctime>
#include "instructionDefines.h"

namespace instructionParam {
	struct myCudaParam_t {
		float x, y, z, radius, strength;
		int idx;
		SimMap* simMap;
		HANDLE handle;
		void(*callback)();
		float sprinklerStrength, sprinklerRadius, evaporation;
	};
	//returns true if certain duation since the last erosion interaction has passed.
	bool checkInterval();
	//passes GPU parameters and enables pointer getters.
	void passParams(SimMap* simMap, int layerIndex, float strength, float radius, float x, float y, float z,
		float sprinklerStrength, float sprinklerRadius, float evaporation);
	void genSum(SimMap* simMap, int * d_param);
	int* getIntPtr();
	float* getFloatPtr();
	float** getExtraPtr();
}