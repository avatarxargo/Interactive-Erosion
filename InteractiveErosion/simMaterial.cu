#include "simMaterial.cuh"

SimMaterial::SimMaterial() {
	name = "Bedrock (UNINITIALIZED)";
	textureid = 0;
	erosionParams = { 0,0,0,0 };
	passMaterialData();
	//just for array initialization purposes.
}

SimMaterial::SimMaterial(std::string sname, std::string texturePath) {
	name = sname;
	textureid = loadBMP_custom(texturePath.c_str());
	erosionParams = {1,1,30,1};
	passMaterialData();
}

SimMaterial::SimMaterial(std::string sname, std::string texturePath, ErosionParam params) {
	name = sname;
	textureid = loadBMP_custom(texturePath.c_str());
	erosionParams = params;
	passMaterialData();
}

std::string SimMaterial::getName() {
	return name;
}

SimMaterial::ErosionParam SimMaterial::getParams() {
	return erosionParams;
}

SimMaterial::ErosionParam* SimMaterial::getParamsPtr() {
	return &erosionParams;
}

cudaError_t SimMaterial::passMaterialData() {
	const int len = 4;
	float param[len] = { erosionParams.thermalRate, erosionParams.talosAngle, erosionParams.hydroRate, erosionParams.sedimentRate };
	int datasize = len*sizeof(float);
	cudaFree(d_data);
	cudaError_t err = cudaMalloc(&d_data, datasize);
	return cudaMemcpy(d_data, param, datasize, cudaMemcpyHostToDevice);
}

float* SimMaterial::getDataPtr() {
	return d_data;
}

GLuint SimMaterial::getTextureId() {
	return textureid;
}

void SimMaterial::cleanUp() {
	cudaFree(d_data);
}

SimMaterial::~SimMaterial() {
	//nothing really.
}