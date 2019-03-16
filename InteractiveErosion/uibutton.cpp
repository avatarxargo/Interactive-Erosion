#include "uibutton.h"
#include "tools.h"

UIButton::UIButton(int id, UISprite mysprite) {
	sprite = mysprite;
	this->id = id;
	clicked = false;
}

int UIButton::getId() {
	return id;
}

bool UIButton::isMouseOver(float inx, float iny, float uiratio, bool click) {
	float glw = (float)glutGet(GLUT_WINDOW_WIDTH);
	float glh = (float)glutGet(GLUT_WINDOW_HEIGHT);
	//recalculate the expected UI width.
	float myw = ((float)glh) / uiratio;
	float woffset = (glw - myw) / 2;
	float mx = inx;
	float my = iny - glh/2; //
	if (sprite.alignment == Left) {
		if (click && inside(mx, my, sprite.x*glh, (sprite.x + sprite.w)*glh, (sprite.y - sprite.h)*glh, sprite.y*glh)) {
			clicked = !clicked;
			return true;
		}
		else {
			return false;
		}
	}
	else if (sprite.alignment == Right) {
		if (click && inside(mx, my, (glw + sprite.x*glh), (glw + (sprite.x + sprite.w)*glh), (sprite.y - sprite.h)*glh, sprite.y*glh)) {
			clicked = !clicked;
			return true;
		}
		else {
			return false;
		}
	}
	return false;
}

void UIButton::render(GLuint progid) {
	if (clicked) {
		sprite.srcx = 0.5;
	}
	else {
		sprite.srcx = 0;
	}
	sprite.render(progid);
}

void UIButton::setClicked(bool arg) {
	clicked = arg;
}

UIButton::~UIButton() {

}