#pragma once
#include <GL/glew.h>
#include "alignment.h"

class UISprite
{
private:
	float srcy, srcw, srch;
	GLuint texture;
	void uniformAllign(GLuint progID);
public:
	Alignment alignment;
	float x, y, w, h;
	float srcx;
	void render(GLuint progID);
	UISprite();
	UISprite(Alignment align, GLuint tex, float mx, float my, float mw, float mh);
	UISprite(Alignment align, GLuint tex, float mx, float my, float mw, float mh, float msrcx, float msrcy, float msrcw, float msrch);
	~UISprite();
};