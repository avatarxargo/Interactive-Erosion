#include "tools.h"

void renderPointer(glm::vec3 pointer) {
	glBegin(GL_TRIANGLES);
	glVertex3f(pointer.x, pointer.y, pointer.z);
	glVertex3f(pointer.x, pointer.y + 2, pointer.z + 1);
	glVertex3f(pointer.x, pointer.y + 2, pointer.z - 1);
	glVertex3f(pointer.x, pointer.y, pointer.z);
	glVertex3f(pointer.x + 1, pointer.y + 2, pointer.z);
	glVertex3f(pointer.x - 1, pointer.y + 2, pointer.z);
	glEnd();
}

GLfloat readDepth(int x, int y) {
	GLfloat data;
	glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &data);
	return data;
}

glm::vec3 dedim(glm::vec4 vec) {
	return glm::vec3(vec.x, vec.y, vec.z);
}

bool inside(float x, float y, float xbot, float xtop, float ybot, float ytop) {
	return x > xbot && x < xtop && y > ybot && y < ytop;
}

bool loadImageInfo(unsigned int* width, unsigned int* height, unsigned int* imageSize, const char* imagepath) {
	// Data read from the header of the BMP file
	unsigned char header[54]; // Each BMP file begins by a 54-bytes header
	unsigned int dataPos;     // Position in the file where the actual data begins
	unsigned char * data;
	float * h_dataHeight;
	// Open the file
	FILE * file = fopen(imagepath, "rb");
	if (file == NULL) {
		// Error, as expected.
		perror("Error opening file");
		printf("Error code opening file: %d\n", errno);
		printf("Error opening file: %s\n", strerror(errno));
		exit(-1);
	}
	if (!file) { printf("Image could not be opened\n"); return false; }

	if (fread(header, 1, 54, file) != 54) { // If not 54 bytes read : problem
		printf("Not a correct BMP file\n");
		return false;
	}

	if (header[0] != 'B' || header[1] != 'M') {
		printf("Not a correct BMP file\n");
		return false;
	}

	// Read ints from the byte array
	dataPos = *(int*)&(header[0x0A]);
	*imageSize = *(int*)&(header[0x22]);
	*width = *(int*)&(header[0x12]);
	*height = *(int*)&(header[0x16]);
}

bool loadRGBdata(float* h_dataHeightALL, float* h_dataHeightR, float* h_dataHeightG, float* h_dataHeightB, float vscale, const char * imagepath)
{
	// Data read from the header of the BMP file
	unsigned char header[54]; // Each BMP file begins by a 54-bytes header
	unsigned int dataPos;     // Position in the file where the actual data begins
	unsigned int width, height;
	unsigned int imageSize;   // = width*height*3
							  // Actual RGB data
	unsigned char * data;
	float * h_dataHeight;
	// Open the file
	FILE * file = fopen(imagepath, "rb");
	if (file == NULL) {
		// Error, as expected.
		perror("Error opening file");
		printf("Error code opening file: %d\n", errno);
		printf("Error opening file: %s\n", strerror(errno));
		exit(-1);
	}
	if (!file) { printf("Image could not be opened\n"); return false; }

	if (fread(header, 1, 54, file) != 54) { // If not 54 bytes read : problem
		printf("Not a correct BMP file\n");
		return false;
	}

	if (header[0] != 'B' || header[1] != 'M') {
		printf("Not a correct BMP file\n");
		return false;
	}

	// Read ints from the byte array
	dataPos = *(int*)&(header[0x0A]);
	imageSize = *(int*)&(header[0x22]);
	width = *(int*)&(header[0x12]);
	height = *(int*)&(header[0x16]);

	// Some BMP files are misformatted, guess missing information
	if (imageSize == 0)    imageSize = width*height * 3; // 3 : one byte for each Red, Green and Blue component
	if (dataPos == 0)      dataPos = 54; // The BMP header is done that way
										 // Create a buffer
	data = new unsigned char[imageSize];

	int res = fread(data, sizeof(unsigned char), imageSize, file);
	int end = feof(file);
	int err = ferror(file);

	//Everything is in memory now, the file can be closed
	fclose(file);

	//Use buffer to remove 0s at the end of each row. buffer size = (total data len - w*h*3)/h
	unsigned int buffer = (imageSize - width*height * 3) / height;
	//printf("Buffer size for data %u and count %u: %u\n", imageSize, width*height * 3, buffer);
	for (int i = 0; i < height; ++i) {
		unsigned int offset = i*(width + buffer) * 3;
		for (int j = 0; j < width; ++j) {
			//every 3rd element is either R,G or B
			h_dataHeightB[i*width + j] = vscale * (float)data[offset + j * 3]; //every 3rd element is either R,G or B
			h_dataHeightG[i*width + j] = vscale * (float)data[offset + j * 3 + 1]; //every 3rd element is either R,G or B
			h_dataHeightR[i*width + j] = vscale * (float)data[offset + j * 3 + 2]; //every 3rd element is either R,G or B
			h_dataHeightALL[i*width + j] = h_dataHeightB[i*width + j] + h_dataHeightG[i*width + j] + h_dataHeightR[i*width + j];
		}
	}
	free(data);
}