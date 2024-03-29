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
#include "instructionParam.cuh"
#include "instructionDefines.h"

namespace hydro {
	/**updates water.*/
	void simWater(float x, float y, float z, bool kinetic, bool hydraulic, int idx, float toolStregth, float toolRadius, float dir, SimMap* simMap, HANDLE handle, void(*callback)(), float sprinklerStrength, float  sprinklerRadius, float evaporation);
	//
	void initThread();
	void killThread();
}