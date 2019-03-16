#ifndef __loadbmp_h__
#define __loadbmp_h__

#include <CImg.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>

//reads the image and creates a GL buffer for it.
GLuint loadBMP_custom(const char * imagepath);

#endif