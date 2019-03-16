#include "uisprite.h"
#include <stdio.h>

UISprite::UISprite() {
	//meh
}

void UISprite::uniformAllign(GLuint progID) {
	switch (alignment) {
	case Left: glUniform1i(glGetUniformLocation(progID, "allign"), 0); return;
	case UpLeft: glUniform1i(glGetUniformLocation(progID, "allign"), 1); return;
	case DownLeft: glUniform1i(glGetUniformLocation(progID, "allign"), 2); return;
	case Right: glUniform1i(glGetUniformLocation(progID, "allign"), 3); return;
	case UpRight: glUniform1i(glGetUniformLocation(progID, "allign"), 4); return;
	case DownRight: glUniform1i(glGetUniformLocation(progID, "allign"), 5); return;
	case Up: glUniform1i(glGetUniformLocation(progID, "allign"), 6); return;
	case Down: glUniform1i(glGetUniformLocation(progID, "allign"), 7); return;
	case ScreenLeft: glUniform1i(glGetUniformLocation(progID, "allign"), 8); return;
	case ScreenUpLeft: glUniform1i(glGetUniformLocation(progID, "allign"), 9); return;
	case ScreenDownLeft: glUniform1i(glGetUniformLocation(progID, "allign"), 10); return;
	case ScreenRight: glUniform1i(glGetUniformLocation(progID, "allign"), 11); return;
	case ScreenUpRight: glUniform1i(glGetUniformLocation(progID, "allign"), 12); return;
	case ScreenDownRight: glUniform1i(glGetUniformLocation(progID, "allign"), 13); return;
	case Centre: glUniform1i(glGetUniformLocation(progID, "allign"), 14); return;
	}
}

void UISprite::render(GLuint progID) {
	//set texture specific uniforms
	glUniform1f(glGetUniformLocation(progID, "x"), x);
	glUniform1f(glGetUniformLocation(progID, "y"), y);
	glUniform1f(glGetUniformLocation(progID, "w"), w);
	glUniform1f(glGetUniformLocation(progID, "h"), h);
	glUniform1f(glGetUniformLocation(progID, "srcx"), srcx);
	glUniform1f(glGetUniformLocation(progID, "srcy"), srcy);
	glUniform1f(glGetUniformLocation(progID, "srcw"), srcw);
	glUniform1f(glGetUniformLocation(progID, "srch"), srch);
	uniformAllign(progID);
	glBindTexture(GL_TEXTURE_2D, texture);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

UISprite::UISprite(Alignment align, GLuint tex, float mx, float my, float mw, float mh) {
	x = mx;
	y = my;
	w = mw;
	h = mh;
	srcx = 0;
	srcy = 0;
	srcw = 1;
	srch = 1;
	alignment = align;
	texture = tex;
}

UISprite::UISprite(Alignment align, GLuint tex, float mx, float my, float mw, float mh, float msrcx, float msrcy, float msrcw, float msrch) {
	x = mx;
	y = my;
	w = mw;
	h = mh;
	srcx = msrcx;
	srcy = msrcy;
	srcw = msrcw;
	srch = msrch;
	alignment = align;
	texture = tex;
}

UISprite::~UISprite() {

}