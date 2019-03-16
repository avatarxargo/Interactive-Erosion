#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <windows.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <GL/freeglut.h>

void renderPointer(glm::vec3 pointer);

GLfloat readDepth(int x, int y);

glm::vec3 dedim(glm::vec4 vec);

bool inside(float x, float y, float xbot, float xtop, float ybot, float ytop);

/**Reads the header of a BMP image file.*/
bool loadImageInfo(unsigned int* width, unsigned int* height, unsigned int* imageSize, const char* imagepath);

/**Loads data from an RGB image.*/
bool loadRGBdata(float* h_dataHeightALL,float* h_dataHeightR, float* h_dataHeightG, float* h_dataHeightB, float vscale, const char * imagepath);