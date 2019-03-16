#pragma once
#include <Cimg.h>
#include <vector>
#include <algorithm>
#include <windows.h>
#include <mutex>
#include <string>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include "cuda.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "simLayer.cuh"
#include "simMaterial.cuh"
#include "polygon.cuh"
#include "tools.h"

//Container for a number of SimLayers
class SimMap
{
private:
	//pointer to the array of data layers on the GPU. 
	std::vector<SimLayer> layers;
	std::vector<SimMaterial> materials;
	SimPolygon polySelect;
	SimPolygon waterSprinkler;
	//maxslope is actually the sin of the max slope angle, since that is easier to compute from the data and is the required value for hydraulic erosion.
	SimLayer selectionMask, waterLevel, waterFlow, waterLast, particles,
		sedimentation, soilHydrataion, waterFlowVert, waterFlowHor,
		waterCellFlowVert, waterCellFlowHor, sum, maxslope;
	int width, height, maxLayers;
	float** h_layerDataList;
	//pointer to an array of pointers directly to layer data arrays in device memory.
	float** d_layerDataList;
	//passes the host list of material material indeces pointers to the device.
	int* h_layerMatIndexList;
	//pointer to an array of pointers directly to layer material indeces arrays in device memory.
	int* d_layerMatIndexList;
	//passes the host list of material data pointers to the device.
	float** h_materialDataList;
	//pointer to an array of pointers directly to material data arrays in device memory.
	float** d_materialDataList;
	HANDLE simMutex;
public:
	SimMap();
	SimMap(int width, int height, int maxLayers);
	void init(int width, int height, int maxLayers);
	/**Creates a new inner layer and allocates memory for it in the device.*/
	void addLayer(int idx, std::string name);
	void addLayer(int idx, std::string name, int materialIdx);
	void addMaterial(int idx, std::string name, std::string texturePath, SimMaterial::ErosionParam params);
	SimLayer* getLayer(int idx);
	SimMaterial* getMaterial(int idx);
	/**Actually inserts height data into the texture.*/
	void setLayerData(int idx, float * h_initData);
	void setLayerDataFromFile(const char * path, int idx, bool r, bool g, bool b, float scale);
	void removeLayer();
	void removeLayer(int idx);
	void removeMaterial(int idx);
	//removes all existing layers and cleans up their GPU memory.
	void cleanseExistingLayers();
	bool containsMaterial(std::string name);
	int getLayerCount();
	void passPolygons();
	//passes the host list of material data pointers to the device.
	void passMaterialListPointers();
	//passes the host list of layer data pointers to the device.
	void passLayerListPointers();
	float** getDeviceLayerDataList();
	int* getDeviceLayerMaterialIndexList();
	float** getDeviceMaterialDataList();
	float* getMaskPtr();
	float* getWaterPtr();
	int getWidth();
	int getHeight();
	int getMaterialCount();
	SimPolygon* getPoly();
	SimPolygon* getSprinkler();
	//for tweak bar
	void __stdcall callbackAdd(void *clientData, int idx, int rgb, float scale);
	void __stdcall callbackAddMaterial(void *clientData, int idx);
	void callbackAddBlank(int idx);
	void callbackRemove(int idx);
	void callbackRemoveMaterial(int idx);
	void callbackMove(int idx, bool up);
	void setMutex(HANDLE ptr);
	HANDLE getMutex();
	~SimMap();
};
