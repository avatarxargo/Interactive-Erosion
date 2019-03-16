#include "polygon.cuh"

void SimPolygon::resetPoly() {
	poly.clear();
	finished = false;
}

glm::vec3 SimPolygon::getPoint(int idx) {
	return poly[idx];
}

std::vector<glm::vec3> SimPolygon::getVector() {
	return poly;
}

void SimPolygon::deleteData() {
	cudaFree(d_poly);
}

void SimPolygon::passData() {
	size_t siz = poly.size() * 3 * sizeof(float);
	h_poly = (float*)malloc(siz);
	for (unsigned int i = 0; i < poly.size(); ++i) {
		h_poly[i * 3] = poly[i].x;
		h_poly[i * 3 + 1] = poly[i].y;
		h_poly[i * 3 + 2] = poly[i].z;
	}
	cudaError_t err = cudaFree(d_poly);
	if (err != 0)
	printf("(1) CUDA PASS ERR: %d\n", err);
	err = cudaMalloc(&d_poly, siz);
	if (err != 0)
	printf("(2) CUDA PASS ERR: %d\n", err);
	err = cudaMemcpy(d_poly, h_poly, siz, cudaMemcpyHostToDevice);
	if (err != 0)
	printf("(3) CUDA PASS ERR: %d\n", err);
	free(h_poly);
	cudaStreamSynchronize(0);
}

float * SimPolygon::getDeviceDataPtr() {
	return d_poly;
}

int SimPolygon::getPointByPosition(float x, float y, float sensitivity) {
	for (unsigned int i = 0; i < poly.size(); ++i) {
		if ((x-poly[i].x)*(x - poly[i].x) + (y - poly[i].z)*(y - poly[i].z) < sensitivity*sensitivity) {
			return i;
		}
	}
	return -1;
}

void SimPolygon::setPoint(int idx, glm::vec3 pos) {
	poly[idx] = pos;
}

void SimPolygon::deletePoint(int idx) {
	poly.erase(poly.begin()+idx);
}

void SimPolygon::shiftPoints() {
	for (unsigned int i = 0; i < poly2add.size(); ++i) {
		poly.push_back(poly2add[i]);
	}
	poly2add.clear();
}

void SimPolygon::addPoint(glm::vec3 pos) {
	poly2add.push_back(pos);
}

SimPolygon::SimPolygon() {

}

SimPolygon::~SimPolygon() {

}