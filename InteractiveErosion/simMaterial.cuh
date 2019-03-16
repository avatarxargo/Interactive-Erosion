#pragma once
#include <windows.h>
#include "loadbmp.h"
#include "cuda.h"
#include <string>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

class SimMaterial
{
public:
	struct ErosionParam {
		/**Rate of release in thermal simulation*/
		float thermalRate;
		/**Cutoff angle for when the material becomes stable*/
		float talosAngle;
		/**Rate of release in hydro simulation*/
		float hydroRate;
		/**Rate of sedimentation*/
		float sedimentRate;
	};
private:
	GLuint textureid;
	ErosionParam erosionParams;
	float* d_data;
public:
	std::string name;
	ErosionParam getParams();
	ErosionParam* SimMaterial::getParamsPtr();
	/**destorys the GL resource associated with this material.*/
	void cleanUp();
	/**Moves the material data onto GPU.*/
	cudaError_t passMaterialData();
	float* getDataPtr();
	GLuint getTextureId();
	std::string SimMaterial::getName();
	SimMaterial();
	SimMaterial(std::string sname, std::string texturePath);
	SimMaterial(std::string sname, std::string texturePath, ErosionParam params);
	~SimMaterial();
};