#pragma once
#include <vector>
#include "uisprite.h"
#include "uibutton.h"

class UISystem
{
private:
	GLuint uishader;
	GLuint uivao, uivbo, uieab;
	float uiratio;
	std::vector<UIButton> buttons;
	std::vector<UISprite> sprites;
public:
	void render();
	int isUIOver(float x, float y, bool click);
	void deactivateOther(int idx);
	void genArrays(GLuint shader);
	void addSprite(UISprite sprite);
	void addButton(UIButton button);
	UIButton getButton(int idx);
	UISystem();
	UISystem(float ratio);
	~UISystem();
};