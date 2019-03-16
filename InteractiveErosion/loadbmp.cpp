#include "loadbmp.h"

GLuint loadBMP_custom(const char * imagepath) {
	cimg_library::CImg<unsigned char> img(imagepath);

	GLuint textureID;
	glGenTextures(1, &textureID);

	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, textureID);

	//process data into the correct format
	int sw = img.width();
	int sh = img.height();
	img.permute_axes("cxzy");

	// Give the image to OpenGL
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sw, sh, 0, GL_RGB, GL_UNSIGNED_BYTE, img);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	return textureID;
}