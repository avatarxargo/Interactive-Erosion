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

namespace pass {
	/**passes indexes, layer and material data to device.*/
	void passData(SimMap* simMap, HANDLE handle);
	//
	void initThread();
	void killThread();
}