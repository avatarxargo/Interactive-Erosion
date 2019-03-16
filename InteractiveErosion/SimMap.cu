#include "simmap.cuh"
#include "call.cuh"

SimMap::SimMap() {
	//no device memory allocated until init is called during runtime.
	width = 0;
	height = 0;
	maxLayers = 0;
}

SimMap::SimMap(int iheight = 1, int iwidth = 1, int imaxLayers = 1) {
	init(iheight,iwidth,imaxLayers);
}

void SimMap::init(int iwidth, int iheight, int imaxLayers) {
	//set major constants
	width = iwidth;
	height = iheight;
	maxLayers = imaxLayers;
	//
	selectionMask = SimLayer(width, height, "mask");
	waterLevel = SimLayer(width, height, "water");
	waterLast = SimLayer(width, height, "waterLast");
	particles = SimLayer(width, height, "particles");
	sedimentation = SimLayer(width, height, "sedimentation");
	soilHydrataion = SimLayer(width, height, "soilHydrataion");
	waterFlowVert = SimLayer(width, height, "waterFlowVert");
	waterFlowHor = SimLayer(width, height, "waterFlowHor");
	waterCellFlowVert = SimLayer(width, height, "waterCellFlowVert");
	waterCellFlowHor = SimLayer(width, height, "waterCellFlowHor");
	maxslope = SimLayer(width, height, "Max Slope");
	sum = SimLayer(width, height, "sum");
	selectionMask.blankData(1);
	waterLevel.blankData(0);
	waterLast.blankData(0);
	particles.blankData(0);
	sedimentation.blankData(0);
	soilHydrataion.blankData(0);
	waterFlowVert.blankData(0);
	waterFlowHor.blankData(0);
	waterCellFlowVert.blankData(0);
	waterCellFlowHor.blankData(0);
	maxslope.blankData(0);
	sum.blankData(0);
}

void SimMap::passLayerListPointers() {
	passPolygons();
	free(h_layerDataList);
	free(h_layerMatIndexList);
	int sizer = layers.size() + 14;
	h_layerDataList = new float*[sizer];
	h_layerMatIndexList = new int[layers.size()];
	//get data pointers up to date.
	h_layerDataList[layers.size()] = waterLevel.getDataPtr();
	h_layerDataList[layers.size()+1] = selectionMask.getDataPtr();
	h_layerDataList[layers.size()+2] = sum.getDataPtr();
	h_layerDataList[layers.size() + 3] = waterLast.getDataPtr();
	h_layerDataList[layers.size() + 4] = particles.getDataPtr();
	h_layerDataList[layers.size() + 5] = sedimentation.getDataPtr();
	h_layerDataList[layers.size() + 6] = soilHydrataion.getDataPtr();
	h_layerDataList[layers.size() + 7] = waterFlowVert.getDataPtr();
	h_layerDataList[layers.size() + 8] = waterFlowHor.getDataPtr();
	h_layerDataList[layers.size() + 9] = waterCellFlowVert.getDataPtr();
	h_layerDataList[layers.size() + 10] = waterCellFlowHor.getDataPtr();
	h_layerDataList[layers.size() + 11] = maxslope.getDataPtr();
	h_layerDataList[layers.size() + 12] = polySelect.getDeviceDataPtr();
	h_layerDataList[layers.size() + 13] = waterSprinkler.getDeviceDataPtr();
	for (int i = layers.size()-1; i >= 0; --i) {
		h_layerDataList[i] = layers[i].getDataPtr();
		h_layerMatIndexList[i] = layers[i].getMaterialIdx();
	}
	//copy the pointers to device.
	size_t datasize = sizeof(float*)*(sizer);
	size_t datasizeMat = sizeof(int)*layers.size();
	cudaDeviceSynchronize();
	cudaError_t err = cudaFree(d_layerDataList);
	err = cudaMalloc(&d_layerDataList, datasize);
	err = cudaMemcpy(d_layerDataList, h_layerDataList, datasize, cudaMemcpyHostToDevice);
	//
	err = cudaFree(d_layerMatIndexList);
	err = cudaMalloc(&d_layerMatIndexList, datasizeMat);
	err = cudaMemcpy(d_layerMatIndexList, h_layerMatIndexList, datasizeMat, cudaMemcpyHostToDevice);
}

void SimMap::passPolygons() {
	polySelect.passData();
	waterSprinkler.passData();
}

void SimMap::passMaterialListPointers() {
	free(h_materialDataList);
	h_materialDataList = new float*[materials.size()];
	//get data pointers up to date.
	for (int i = 0; i < materials.size(); ++i) {
		materials[i].passMaterialData();
		h_materialDataList[i] = materials[i].getDataPtr();
	}
	//copy the pointers to device.
	int datasize = sizeof(float*)*materials.size();
	cudaFree(d_materialDataList);
	cudaMalloc(&d_materialDataList, datasize);
	cudaMemcpy(d_materialDataList, h_materialDataList, datasize, cudaMemcpyHostToDevice);
}

void SimMap::setLayerData(int idx, float* data) {
	int ret = layers[idx].setData(data);
	passLayerListPointers();
}

void SimMap::addMaterial(int idx, std::string name, std::string texturePath, SimMaterial::ErosionParam params) {
	//fix indeces
	for (int i = 0; i < layers.size(); ++i) {
		if (layers[i].getMaterialIdx() >= idx) {
			//increase the higher or equal layers by 1.
			layers[i].setMaterialIdx(layers[i].getMaterialIdx() + 1);
		}
	}
	SimMaterial material(name, texturePath, params);
	materials.insert(materials.begin() + idx, material);
	//update the gpu records.
	passMaterialListPointers();
}

void SimMap::addLayer(int idx, std::string name) {
	SimLayer layer(width, height, name);
	layers.insert(layers.begin()+idx, layer);
}

void SimMap::addLayer(int idx, std::string name, int materialIdx) {
	SimLayer layer(width, height, name, materialIdx);
	layers.insert(layers.begin() + idx, layer);
}

SimLayer* SimMap::getLayer(int idx) {
	return &layers[idx];
}

SimMaterial* SimMap::getMaterial(int idx) {
	return &materials[idx];
}

bool SimMap::containsMaterial(std::string name) {
	for (int i = 0; i < materials.size(); ++i) {
		if (name.compare(materials[i].name) == 0) {
			return true;
		}
	}
	return false;
}

void SimMap::removeLayer() {
	layers.pop_back();
	passLayerListPointers();
}

void SimMap::removeLayer(int idx) {
	layers[idx].cleanUp();
	layers.erase(layers.begin() + idx);
	passLayerListPointers();
}

void SimMap::removeMaterial(int idx) {
	for (int i = 0; i < layers.size(); ++i) {
		if (layers[i].getMaterialIdx() > idx) {
			//decrease the higher layers by 1.
			layers[i].setMaterialIdx(layers[i].getMaterialIdx()-1);
		} else if (layers[i].getMaterialIdx() == idx) {
			//reset all layers using this material to 0
			layers[i].setMaterialIdx(0);
		}
	}
	materials[idx].cleanUp();
	materials.erase(materials.begin() + idx);
	passMaterialListPointers();
}

int SimMap::getLayerCount() {
	return (int)layers.size();
}

int SimMap::getWidth() {
	return width;
}
int SimMap::getHeight() {
	return height;
}

SimPolygon* SimMap::getPoly() {
	return &polySelect;
}

SimPolygon* SimMap::getSprinkler() {
	return &waterSprinkler;
}

int SimMap::getMaterialCount() {
	return materials.size();
}

float** SimMap::getDeviceLayerDataList() {
	return d_layerDataList;
}

int* SimMap::getDeviceLayerMaterialIndexList() {
	return d_layerMatIndexList;
}

float** SimMap::getDeviceMaterialDataList() {
	return d_materialDataList;
}

float * SimMap::getMaskPtr() {
	return waterLevel.getDataPtr();
}

float * SimMap::getWaterPtr() {
	return selectionMask.getDataPtr();
}

void SimMap::setMutex(HANDLE handle) {
	simMutex = handle;
}

HANDLE SimMap::getMutex() {
	return simMutex;
}

void SimMap::callbackMove(int idx, bool up) {
	if (up) {
		iter_swap(layers.begin() + idx, layers.begin() + idx + 1);
	}
	else {
		iter_swap(layers.begin() + idx, layers.begin() + idx - 1);
	}
	passLayerListPointers();
}

void SimMap::setLayerDataFromFile(const char * path, int idx, bool r, bool g, bool b, float scale){
	//generate data
	float * data = new float[width*height];
	cimg_library::CImg<unsigned char> img(path);
	int offset = img.height()*img.width();
	for (unsigned int i = 0; i < height; ++i) {
		for (unsigned int j = 0; j < width; ++j) {
			int clampi = i % img.height();
			int clampj = j % img.width();
			data[width * i + j] = 0;
			if (r)
				data[width * i + j] += img[ img.width() * clampi + clampj] * scale/255.0f;
			if (g)
				data[width * i + j] += img[ offset + img.width() * clampi + clampj] * scale / 255.0f;
			if (b)
				data[width * i + j] += img[ offset * 2 + img.width() * clampi + clampj] * scale / 255.0f;
		}
	}
	//
	this->setLayerData(idx, data);
	free(data);
}

void __stdcall  SimMap::callbackAdd(void *clientData, int idx, int rgb, float scale) {
	if (layers.size() >= maxLayers) { return; }
	std::string bmp_file = "data/";
	bmp_file = bmp_file+(char *)clientData;
	//generate data
	//
	std::string layerName = (char *)clientData;
	if (layerName.find(".") != std::string::npos) {
		layerName = layerName.substr(0, layerName.find_last_of('.'));
	}
	this->addLayer(idx, layerName);
	//
	setLayerDataFromFile(bmp_file.c_str(), idx, rgb == 0 || rgb == 1, rgb == 0 || rgb == 2, rgb == 0 || rgb > 2, scale);
	passLayerListPointers();
}

void SimMap::cleanseExistingLayers() {
	for (int i = 0; i < layers.size(); ++i) {
		layers[i].cleanUp();
	}
	layers.clear();
	cudaFree(d_layerDataList);
}

void SimMap::callbackAddBlank(int idx) {
	std::string layerName = "blank layer";
	this->addLayer(idx, layerName, 0);
	layers[idx].blankData(0);
	passLayerListPointers();
}

void __stdcall SimMap::callbackAddMaterial(void *clientData, int idx) {
	std::string bmp_file = "tex/";
	bmp_file = bmp_file + (char *)clientData;
	//
	std::string matName = "Material ";
	matName += std::to_string((int)materials.size());
	//
	SimMaterial::ErosionParam params = {1,15,1,1};
	//
	this->addMaterial(materials.size(), matName, bmp_file, params);
}

void SimMap::callbackRemove(int idx) {
	removeLayer(idx);
}

void SimMap::callbackRemoveMaterial(int idx) {
	removeMaterial(idx);
}

SimMap::~SimMap() {
	for (int i = 0; i < layers.size(); ++i) {
		layers[i].cleanUp();
	}
	cudaFree(d_layerDataList);
	printf(">>> Freeing SimMap with up to %d layers.\n", maxLayers);
	delete[] h_layerDataList;
	layers.clear();
}