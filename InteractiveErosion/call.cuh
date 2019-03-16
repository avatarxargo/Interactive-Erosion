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

namespace call {
	/**sums up heights in each layer. Layers passed as an array of layer locations stored within GPU memory.*/
	void passShell(struct cudaGraphicsResource* shellInterlop, struct cudaGraphicsResource* wallInterlop,
		struct cudaGraphicsResource* maskInterlop, struct cudaGraphicsResource* texmaskInterlop,
		struct cudaGraphicsResource* waterInterlop, struct cudaGraphicsResource* waterMaskInterlop,
		struct cudaGraphicsResource* sedimentMaskInterlop,
		SimMap* simMap, int visiblelayers, bool showAll, float scale);
}