#ifndef _DISPLAY_H_
#define _DISPLAY_H_
#include <GL/glew.h>
#include <vector>
#include "camera3d.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <GL/gl.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#define NOMINMAX
#include "SimMap.cuh"

namespace display {

	//prepares all the OpenGL buffers for rendering
	void genDisplayElements(unsigned int width, unsigned int height, unsigned int depth, float scale);

	//render the terrain after it has been updated from the logic functions
	void renderTerrain();

	struct cudaGraphicsResource* getShellInterlop();
	struct cudaGraphicsResource* getWallInterlop();
	struct cudaGraphicsResource* getMaskInterlop();
	struct cudaGraphicsResource* getTexInterlop();
	struct cudaGraphicsResource* getWaterInterlop();
	struct cudaGraphicsResource* getWaterMaskInterlop();
	struct cudaGraphicsResource* getSedimentMaskInterlop();

	void loadShaders();

	GLuint getUIShader();
	GLuint getTerrainShader();
	GLuint getWaterShader();

	float getScale();

	void genMask();

	GLuint LoadShader(const char * vertex_file_path, const char * fragment_file_path);

	void displayCall(Camera3D camera, int visibleLayers, glm::vec3 pointer, glm::mat4 *view, glm::mat4 *proj, glm::mat4 *model, float toolRadius, int displayMode, int tool, SimMap* simMap, float farv);
}
#endif //_DISPLAY_H_