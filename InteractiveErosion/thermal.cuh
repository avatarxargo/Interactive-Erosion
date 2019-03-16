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
#include "math.h"
#include "instructionParam.cuh"
#include "instructionDefines.h"
#include <condition_variable>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace thermal {
	/**Conducts thermal erosion on the provided SimMap.*/
	void erodeThermal(float x, float y, float z, int idx, float toolStregth, float toolRadius, float dir, SimMap* simMap, HANDLE handle, void(*callback)(), float sprinklerStrength, float  sprinklerRadius, float evaporation);
	//
	void initThread();
	void killThread();
}