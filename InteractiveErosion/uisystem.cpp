#include "uisystem.h"
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <freeglut.h>

UISystem::UISystem() {
	uiratio = 3.0 / 4.0;
}

UISystem::UISystem(float ratio = 3.0 / 4.0) {
	uiratio = ratio;
}

void UISystem::genArrays(GLuint shader) {
	uishader = shader;
	GLfloat vertices_position[24] = {
		0, 0,
		1, 0,
		1, 1,
		0, 1
	};
	GLfloat texture_coord[8] = {
		0.0, 0.0,
		1.0, 0.0,
		1.0, 1.0,
		0.0, 1.0
	};
	GLuint indices[6] = {
		0, 1, 2,
		2, 3, 0
	};

	glGenVertexArrays(1, &uivao);
	glBindVertexArray(uivao);

	glGenBuffers(1, &uieab);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uieab);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glGenBuffers(1, &uivbo);
	glBindBuffer(GL_ARRAY_BUFFER, uivbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_position) + sizeof(texture_coord), vertices_position, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(vertices_position), sizeof(texture_coord), texture_coord);

	GLint position_attribute = glGetAttribLocation(uishader, "position");
	glVertexAttribPointer(position_attribute, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(position_attribute);

	//texture
	GLint texture_coord_attribute = glGetAttribLocation(uishader, "texture_coord");
	glVertexAttribPointer(texture_coord_attribute, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid *)sizeof(vertices_position));
	glEnableVertexAttribArray(texture_coord_attribute);
	glBindVertexArray(0);
}

void UISystem::render() {
	float w = glutGet(GLUT_WINDOW_WIDTH);
	float h = glutGet(GLUT_WINDOW_HEIGHT);

	glDisable(GL_DEPTH_TEST);
	glUseProgram(uishader);
	//
	glm::mat4 Model, View, Projection;
	Projection = glm::ortho(-w / h, w / h, -1.0f, 1.0f, -1.0f, 1.0f);
	// Transfer the transformation matrices to the shader program
	GLint model = glGetUniformLocation(uishader, "Model");
	glUniformMatrix4fv(model, 1, GL_FALSE, glm::value_ptr(Model));

	GLint view = glGetUniformLocation(uishader, "View");
	glUniformMatrix4fv(view, 1, GL_FALSE, glm::value_ptr(View));

	GLint projection = glGetUniformLocation(uishader, "Projection");
	glUniformMatrix4fv(projection, 1, GL_FALSE, glm::value_ptr(Projection));
	//
	float expw = glutGet(GLUT_WINDOW_HEIGHT) / uiratio;
	float Shift = ((glutGet(GLUT_WINDOW_WIDTH) - expw)) / 2;
	//printf("Shift: %f w %d exw %f\n", Shift, glutGet(GLUT_WINDOW_WIDTH), expw);
	glUniform1f(glGetUniformLocation(uishader, "Shift"), Shift / glutGet(GLUT_WINDOW_WIDTH));
	//
	glBindVertexArray(uivao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uieab);
	glBindBuffer(GL_ARRAY_BUFFER, uivbo);
	for (unsigned int i = 0; i < sprites.size(); ++i) {
		sprites[i].render(uishader);
	}
	for (unsigned int i = 0; i < buttons.size(); ++i) {
		buttons[i].render(uishader);
	}
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
}

void UISystem::deactivateOther(int idx) {
	for (unsigned int i = 0; i < buttons.size(); ++i) {
		if (i!=idx) {
			buttons[i].clicked = false;
		}
	}
}

int UISystem::isUIOver(float x, float y, bool click) {
	for (unsigned int i = 0; i < buttons.size(); ++i) {
		if (buttons[i].isMouseOver(x, y, uiratio, click)) {
			return i;
		}
	}
	return -1;
}

void UISystem::addSprite(UISprite sprite) {
	sprites.insert(sprites.end(),sprite);
}

void UISystem::addButton(UIButton button) {
	buttons.insert(buttons.end(), button);
}

UIButton UISystem::getButton(int idx) {
	return buttons[idx];
}


UISystem::~UISystem() {

}