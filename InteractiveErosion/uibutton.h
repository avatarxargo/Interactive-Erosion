#pragma once
#include "uisprite.h";

class UIButton
{
private:
	UISprite sprite;
	int id;
public:
	bool isMouseOver(float x, float y, float uiratio, bool clicked);
	bool clicked;
	void render(GLuint progID);
	void setClicked(bool arg);
	int getId();
	UIButton(int id, UISprite sprite);
	~UIButton();
};
