#include "display.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <GL/freeglut.h>
#include <string>
#include <algorithm>

namespace display {
	//vao for the terrain shell
	GLuint terrain_shell_vbo, terrain_shell_ebo, terrain_wall_vbo, terrain_wall_ebo,
		terrain_wall_mask_vbo, terrain_mask_vbo, terrain_tex_vbo, terrain_water_vbo, terrain_water_ebo, water_mask_vbo, sediment_mask_vbo;
	struct cudaGraphicsResource *terrain_shell_vbo_interlop, *terrain_wall_vbo_interlop, *terrain_mask_vbo_interlop,
		*terrain_tex_vbo_interlop, *terrain_water_vbo_interlop, *water_mask_vbo_interlop, *sediment_mask_vbo_interlop;
	int mapWidth, mapHeight, mapDepth = 20;
	float mapScale;
	GLuint progID, progWallID, uiprogID, progWaterID;

	float getScale() {
		return mapScale;
	}

	struct cudaGraphicsResource* getShellInterlop() {
		return terrain_shell_vbo_interlop;
	}

	struct cudaGraphicsResource* getWallInterlop() {
		return terrain_wall_vbo_interlop;
	}

	struct cudaGraphicsResource* getMaskInterlop() {
		return terrain_mask_vbo_interlop;
	}

	struct cudaGraphicsResource* getTexInterlop() {
		return terrain_tex_vbo_interlop;
	}

	struct cudaGraphicsResource* getWaterInterlop() {
		return terrain_water_vbo_interlop;
	}

	struct cudaGraphicsResource* getWaterMaskInterlop() {
		return water_mask_vbo_interlop;
	}

	struct cudaGraphicsResource* getSedimentMaskInterlop() {
		return sediment_mask_vbo_interlop;
	}

	void renderTerrain() {
		//Here the shaders, uniforms etc are set up and we just bind our verteces and draw elements without a care in the world.
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_shell_ebo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_shell_vbo);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
		glEnableVertexAttribArray(0);
		glDrawElements(GL_TRIANGLE_STRIP, mapWidth*(mapHeight - 1) * 2, GL_UNSIGNED_INT, 0);
		//unbind for good measure.
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void renderWater() {
		//Here water has already set up the shaders, uniforms etc and we just bind our verteces and draw elements without a care in the world.
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_water_ebo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_water_vbo);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
		glEnableVertexAttribArray(0);
		glDrawElements(GL_TRIANGLE_STRIP, mapWidth*(mapHeight - 1) * 2, GL_UNSIGNED_INT, 0);
		//unbind for good measure.
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void renderWalls(int visibleLayers) {
		//Here water has already set up the shaders, uniforms etc and we just bind our verteces and draw elements without a care in the world.
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_wall_ebo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_wall_vbo);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0   // array buffer offset
		);
		glEnableVertexAttribArray(0);
		glDrawElements(GL_TRIANGLE_STRIP, (visibleLayers *  ((mapWidth + mapHeight - 2) * 4)), GL_UNSIGNED_INT, 0);
		//unbind for good measure.
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void loadShaders() {
		progID = LoadShader("terrain.vert", "terrain.frag");
		progWallID = LoadShader("wall.vert", "wall.frag");
		progWaterID = LoadShader("water.vert", "water.frag");
		uiprogID = LoadShader("ui.vert", "ui.frag");
	}

	GLuint getUIShader() {
		return uiprogID;
	}

	GLuint getTerrainShader() {
		return progID;
	}

	GLuint getWaterShader() {
		return progWaterID;
	}

	void genDisplayElements(unsigned int width, unsigned int height, unsigned int depth, float scale)
	{
		mapWidth = width;
		mapHeight = height;
		mapDepth = depth;
		mapScale = scale;
		glClearColor(0.7f, 0.8f, 0.9f, 1.0f);
		//Generates buffers for rendering indexed triangle strips of points
		GLfloat * g_terrain_shell_vertex;
		GLfloat * g_terrain_wall_vertex;
		g_terrain_shell_vertex = new GLfloat[width * height * 3];
		//generate flat terrain to begin with (will be updated once simulation layers exist)
		for (int i = 0; i < height*width * 3; ++i) {
			g_terrain_shell_vertex = 0;
		}
		//generate wall
		int loopLength = 2 * (width + height) - 4;
		int wallVertLen = (depth + 1)*loopLength;
		g_terrain_wall_vertex = new GLfloat[wallVertLen * 3];
		for (int i = 0; i < wallVertLen * 3; ++i) {
			g_terrain_wall_vertex = 0;
		}
		//generate the actual buffers
		glGenBuffers(1, &terrain_shell_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_shell_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*width * height * 3, g_terrain_shell_vertex, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&terrain_shell_vbo_interlop, terrain_shell_vbo,
			cudaGraphicsMapFlagsWriteDiscard);
		//gen walls
		glGenBuffers(1, &terrain_wall_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_wall_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*wallVertLen * 3, g_terrain_wall_vertex, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&terrain_wall_vbo_interlop, terrain_wall_vbo,
			cudaGraphicsMapFlagsWriteDiscard);
		//gen wall material ids
		float* wallmMaterial = new float[wallVertLen];
		for (int t = 0; t < depth+1; ++t) {
			for (int v = 0; v < loopLength; ++v) {
				wallmMaterial[loopLength*t+v] = t;
			}
		}
		glGenBuffers(1, &terrain_wall_mask_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_wall_mask_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*wallVertLen, wallmMaterial, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//gen water too
		glGenBuffers(1, &terrain_water_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_water_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*width * height * 3, g_terrain_shell_vertex, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&terrain_water_vbo_interlop, terrain_water_vbo,
		cudaGraphicsMapFlagsWriteDiscard);

		//generate element pointers
		bool dir = true;
		GLuint * g_terrain_shell_elements;
		g_terrain_shell_elements = new GLuint[2 * width * (height - 1)];
		for (int i = 0; i < height - 1; ++i) {
			//passes each row of faces -> height-1
			int offset = i*width;
			if (dir) {
				for (int j = 0; j < width; j++) {
					g_terrain_shell_elements[(offset + j) * 2] = offset + j;
					g_terrain_shell_elements[(offset + j) * 2 + 1] = offset + width + j;
				}
			}
			else {
				for (int j = 0; j < width; j++) {
					g_terrain_shell_elements[(offset + j) * 2] = offset + width - j - 1;
					g_terrain_shell_elements[(offset + j) * 2 + 1] = offset + 2 * width - j - 1;
				}
			}
			dir = !dir;
		}
		//generate wall element pointers
		GLuint * g_terrain_wall_elements;
		int wallEleLen = 2 * (depth - 1) * loopLength;
		g_terrain_wall_elements = new GLuint[wallEleLen];
		for (int t = 0; t < depth-1; ++t) {
			for (int v = 0; v < loopLength; ++v) {
				int widx = 2 * (loopLength * t + v);
				if (v != loopLength-1) {
					g_terrain_wall_elements[widx] = loopLength*t + v;
					g_terrain_wall_elements[widx + 1] = loopLength*(t + 1) + v;
				}
				else {
					g_terrain_wall_elements[widx] = loopLength*t;
					g_terrain_wall_elements[widx + 1] = loopLength*(t + 1);
				}
			}
		}
		//generate the actual buffers
		glGenBuffers(1, &terrain_shell_ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_shell_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * 2 * width * (height - 1), g_terrain_shell_elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		//generate wall buffers
		glGenBuffers(1, &terrain_wall_ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_wall_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * wallEleLen, g_terrain_wall_elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		//generate water buffers
		glGenBuffers(1, &terrain_water_ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain_water_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * 2 * width * (height - 1), g_terrain_shell_elements, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		//
		free(g_terrain_shell_vertex);
		free(g_terrain_shell_elements);
		free(g_terrain_wall_vertex);
		free(g_terrain_wall_elements);
		free(wallmMaterial);
		genMask();
	}


	void applyProgram(GLuint pid, GLuint mask_vbo, GLuint tex_vbo, Camera3D camera, glm::vec3 pointer, std::vector<glm::vec3>* polygonSelect, std::vector<glm::vec3>* sprinklerList, glm::mat4 *view, glm::mat4 *proj, glm::mat4 *model, float toolRadius, int displayMode, int tool, bool connect, float farv) {
		float w = glutGet(GLUT_WINDOW_WIDTH);
		float h = glutGet(GLUT_WINDOW_HEIGHT);
		glUseProgram(pid);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		//Bind mask
		glBindBuffer(GL_ARRAY_BUFFER, mask_vbo);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(1);
		//
		//Bind tex
		glBindBuffer(GL_ARRAY_BUFFER, tex_vbo);
		glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glEnableVertexAttribArray(2);
		//
		glm::mat4 mview = glm::mat4(1.0f);
		mview = glm::rotate(mview, glm::radians(camera.pitchAngle), glm::vec3(1, 0, 0));
		mview = glm::rotate(mview, glm::radians(camera.yawAngle), glm::vec3(0, 1, 0));
		mview = glm::translate(mview, camera.position);
		glm::mat4 mproj = glm::perspective(glm::radians(45.0f), w / h, 0.1f, farv);
		//
		glm::mat4 mvp = mproj * mview;
		*proj = mproj;
		*view = mview;
		GLint MVP = glGetUniformLocation(pid, "MVP");
		glUniformMatrix4fv(MVP, 1, GL_FALSE, &mvp[0][0]);
		//
		glUniform1i(glGetUniformLocation(pid, "tex1"), 0);
		glUniform1i(glGetUniformLocation(pid, "tex2"), 1);
		//
		glUniform1i(glGetUniformLocation(pid, "renderMode"), displayMode);
		glUniform1i(glGetUniformLocation(pid, "connectPoly"), (tool == 4 || tool == 3) && connect ? 1 : 0);
		glUniform1i(glGetUniformLocation(pid, "showGauss"), tool == 0 || tool == 1 || tool == 7 || tool == 8 ? 1 : 0);
		glUniform1f(glGetUniformLocation(pid, "toolRadius"), toolRadius*mapScale);
		GLint pointerloc = glGetUniformLocation(pid, "pointer");
		glUniform3fv(pointerloc, 1, glm::value_ptr(pointer));
		//polygon
		if (polygonSelect->size() > 0) {
			int i = 0;
			for (; i < polygonSelect->size() && i < 99; i++) {
				std::string locname = "polygon[" + std::to_string(i) + "]";
				GLint polyLoc = glGetUniformLocation(pid, locname.c_str());
				glUniform3f(polyLoc, (*polygonSelect)[i].x, (*polygonSelect)[i].y, (*polygonSelect)[i].z);
			}
			std::string locname = "polygon[" + std::to_string(i) + "]";
			GLint polyLoc = glGetUniformLocation(pid, locname.c_str());
			glUniform3f(polyLoc, pointer.x, pointer.y, pointer.z);
			glUniform1i(glGetUniformLocation(pid, "polygonLength"), std::min((int)polygonSelect->size() + 1, 100));
		}
		else {
			glUniform1i(glGetUniformLocation(pid, "polygonLength"), 0);
		}
		//sprinklers
		if (sprinklerList->size() > 0) {
			for (unsigned int i = 0 ; i < sprinklerList->size() && i < 99; i++) {
				std::string locname = "sprinkler[" + std::to_string(i) + "]";
				GLint sprinkLoc = glGetUniformLocation(pid, locname.c_str());
				glUniform3f(sprinkLoc, (*sprinklerList)[i].x, (*sprinklerList)[i].y, (*sprinklerList)[i].z);
			}
			glUniform1i(glGetUniformLocation(pid, "sprinklerLength"), std::min((int)sprinklerList->size(), 100));
		}
		else {
			glUniform1i(glGetUniformLocation(pid, "sprinklerLength"), 0);
		}
		//textures
		for (int i = 0; i < 10; i++) {
			std::string locname = "texes[" + std::to_string(i) + "]";
			GLint polyLoc = glGetUniformLocation(pid, locname.c_str());
			glUniform1i(polyLoc, i);
		}
	}

	void bindTextures(SimMap* simMap) {
		int count = simMap->getLayerCount();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(0)->getMaterialIdx())->getTextureId());
		if (count < 2) { return; }
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(1)->getMaterialIdx())->getTextureId());
		if (count < 3) { return; }
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(2)->getMaterialIdx())->getTextureId());
		if (count < 4) { return; }
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(3)->getMaterialIdx())->getTextureId());
		if (count < 5) { return; }
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(4)->getMaterialIdx())->getTextureId());
		if (count < 6) { return; }
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(5)->getMaterialIdx())->getTextureId());
		if (count < 7) { return; }
		glActiveTexture(GL_TEXTURE6);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(6)->getMaterialIdx())->getTextureId());
		if (count < 8) { return; }
		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(7)->getMaterialIdx())->getTextureId());
		if (count < 9) { return; }
		glActiveTexture(GL_TEXTURE8);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(8)->getMaterialIdx())->getTextureId());
		if (count < 10) { return; }
		glActiveTexture(GL_TEXTURE9);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(9)->getMaterialIdx())->getTextureId());
		if (count < 11) { return; }
		glActiveTexture(GL_TEXTURE10);
		glBindTexture(GL_TEXTURE_2D, simMap->getMaterial(simMap->getLayer(10)->getMaterialIdx())->getTextureId());
	}

	void displayCall(Camera3D camera, int visibleLayers, glm::vec3 pointer, glm::mat4 *view, glm::mat4 *proj, glm::mat4 *model, float toolRadius, int displayMode, int tool, SimMap* simMap, float farv) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		bindTextures(simMap);
		applyProgram(progID, terrain_mask_vbo, terrain_tex_vbo, camera, pointer, &simMap->getPoly()->getVector(), &simMap->getSprinkler()->getVector(), view, proj, model, toolRadius, displayMode, tool, !simMap->getPoly()->finished,farv);

		// Render the terrain
		renderTerrain();
		applyProgram(progWallID, terrain_wall_mask_vbo, 0, camera, pointer, &simMap->getPoly()->getVector(), &simMap->getSprinkler()->getVector(), view, proj, model, toolRadius, displayMode, tool, !simMap->getPoly()->finished, farv);
		//
		renderWalls(visibleLayers);
		applyProgram(progWaterID, water_mask_vbo, sediment_mask_vbo, camera, pointer, &simMap->getPoly()->getVector(), &simMap->getSprinkler()->getVector(), view, proj, model, toolRadius, displayMode, tool, !simMap->getPoly()->finished, farv);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
		glDepthMask(false);
		renderWater();
		glDepthMask(true);
		//
		glUseProgram(0);
		//
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void genMask() {
		float* mask = new float[mapWidth*mapHeight];
		//gen some data
		for (int i = 0; i < mapWidth; ++i) {
			for (int j = 0; j < mapHeight; ++j) {
				mask[(i*mapWidth + j)] = ((float)i) / mapWidth;
			}
		}
		glGenBuffers(1, &terrain_mask_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_mask_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*mapWidth * mapHeight, mask, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&terrain_mask_vbo_interlop, terrain_mask_vbo,
			cudaGraphicsMapFlagsWriteDiscard);
		//
		glGenBuffers(1, &terrain_tex_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, terrain_tex_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*mapWidth * mapHeight, mask, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&terrain_tex_vbo_interlop, terrain_tex_vbo,
			cudaGraphicsMapFlagsWriteDiscard);
		//
		glGenBuffers(1, &water_mask_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, water_mask_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*mapWidth * mapHeight, mask, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&water_mask_vbo_interlop, water_mask_vbo,
			cudaGraphicsMapFlagsWriteDiscard);
		//
		glGenBuffers(1, &sediment_mask_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, sediment_mask_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*mapWidth * mapHeight, mask, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		cudaGraphicsGLRegisterBuffer(&sediment_mask_vbo_interlop, sediment_mask_vbo,
			cudaGraphicsMapFlagsWriteDiscard);
		free(mask);
	}


	GLuint LoadShader(const char * vertex_file_path, const char * fragment_file_path) {

		// Create the shaders
		GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
		GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
		// Read the Vertex Shader code from the file
		std::string VertexShaderCode;
		std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
		if (VertexShaderStream.is_open()) {
			std::string Line = "";
			while (getline(VertexShaderStream, Line))
				VertexShaderCode += "\n" + Line;
			VertexShaderStream.close();
		}
		else {
			printf("Impossible to open %s. Are you in the correct directory?\n", vertex_file_path);
			getchar();
			return 0;
		}

		// Read the Fragment Shader code from the file
		std::string FragmentShaderCode;
		std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
		if (FragmentShaderStream.is_open()) {
			std::string Line = "";
			while (getline(FragmentShaderStream, Line))
				FragmentShaderCode += "\n" + Line;
			FragmentShaderStream.close();
		}

		GLint Result = GL_FALSE;
		int InfoLogLength;

		// Compile Vertex Shader
		char const * VertexSourcePointer = VertexShaderCode.c_str();
		glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
		glCompileShader(VertexShaderID);

		// Check Vertex Shader
		glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
		glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			printf("SAHDER ERROR\n");
		}
		// Compile Fragment Shader
		char const * FragmentSourcePointer = FragmentShaderCode.c_str();
		glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
		glCompileShader(FragmentShaderID);

		// Check Fragment Shader
		glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
		glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			printf("SAHDER ERROR\n");
		}

		// Link the program
		GLuint ProgramID = glCreateProgram();
		glAttachShader(ProgramID, VertexShaderID);
		glAttachShader(ProgramID, FragmentShaderID);
		glLinkProgram(ProgramID);

		// Check the program
		glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
		glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			printf("SHADER ERROR\n");
		}

		//printf("Cleanup\n");

		glDetachShader(ProgramID, VertexShaderID);
		glDetachShader(ProgramID, FragmentShaderID);

		glDeleteShader(VertexShaderID);
		glDeleteShader(FragmentShaderID);

		return ProgramID;
	}
}