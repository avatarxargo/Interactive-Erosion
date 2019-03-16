#include "simLayer.cuh"

SimLayer::SimLayer() {
	//blank
}

SimLayer::SimLayer(const SimLayer &sl2) {
	materialIdx = sl2.materialIdx;
	width = sl2.width;
	height = sl2.height;
	name = sl2.name;
	d_data = sl2.d_data;
}

SimLayer::SimLayer(int iwidth, int iheight, std::string iname, int sMaterialIdx) {
	materialIdx = sMaterialIdx;
	width = iwidth;
	height = iheight;
	name = iname;
}

SimLayer::SimLayer(int iwidth, int iheight, std::string iname) {
	materialIdx = 0;
	width = iwidth;
	height = iheight;
	name = iname;
	//int datasize = width * height * sizeof(float);
}

cudaError_t SimLayer::setData(float * h_initData) {
	size_t datasize = width * height * sizeof(float);
	cudaFree(d_data);
	cudaError_t err = cudaMalloc(&d_data, datasize);
	return cudaMemcpy(d_data, h_initData, datasize, cudaMemcpyHostToDevice);
}

cudaError_t SimLayer::blankData(float val) {
	float * init = new float[width * height];
	for (int i = 0; i < width*height; ++i) {
		init[i] = val;
	}
	size_t datasize = width * height * sizeof(float);
	cudaFree(d_data);
	cudaError_t err = cudaMalloc(&d_data, datasize);
	err = cudaMemcpy(d_data, init, datasize, cudaMemcpyHostToDevice);
	free(init);
	return err;
}

float* SimLayer::getDataPtr() {
	return d_data;
}

int SimLayer::getMaterialIdx() {
	return materialIdx;
}

void SimLayer::setMaterialIdx(int arg) {
	materialIdx = arg;
}

void SimLayer::cleanUp() {
	cudaFree(d_data);
}

SimLayer::~SimLayer() {
}