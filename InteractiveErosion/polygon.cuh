#pragma once
#include <windows.h>
#include "cuda.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

class SimPolygon
{
private:
	std::vector<glm::vec3> poly;
	std::vector<glm::vec3> poly2add;
	float * h_poly;
	float * d_poly;
public:
	bool drag = false;
	bool finished = false;
	int dragIdx;
	void resetPoly();
	glm::vec3 getPoint(int idx);
	std::vector<glm::vec3> getVector();
	int getPointByPosition(float x, float y, float sensitivity);
	void setPoint(int idx, glm::vec3 pos);
	void addPoint(glm::vec3 pos);
	void shiftPoints();
	void deletePoint(int idx);
	float * getDeviceDataPtr();
	void passData();
	void deleteData();
	SimPolygon();
	~SimPolygon();
};