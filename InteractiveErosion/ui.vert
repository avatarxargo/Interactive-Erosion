#version 330 core
in vec4 position;
in vec2 texture_coord;
out vec2 texture_coord_from_vshader;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform float Shift;
uniform float srcx;
uniform float srcy;
uniform float srcw;
uniform float srch;
uniform float x;
uniform float y;
uniform float w;
uniform float h;
uniform int allign;
uniform float aspect;

void main() {
	if(allign==14) {
		//Allign Center
		gl_Position = (((Projection * View * Model * (position*vec4(w,h,1,1)+vec4(x,-y+0.5,0,0))+vec4(-0.5,-0.5,0,0)))*vec4(2,2,1,1)) + vec4(Shift*2,0,0,0);
	} else if (allign==0) {
		//Allign Left
		gl_Position = (((Projection * View * Model * (position*vec4(w,h,1,1)+vec4(x,-y+0.5,0,0))+vec4(-0.5,-0.5,0,0)))*vec4(2,2,1,1));
	} else { // if (allign==3)
		//Allign Right
		gl_Position = (((Projection * View * Model * (position*vec4(w,h,1,1)+vec4(x+1.33333,-y+0.5,0,0))+vec4(-0.5,-0.5,0,0)))*vec4(2,2,1,1)) + vec4(Shift*4,0,0,0);
	}
	//Texture
	texture_coord_from_vshader = texture_coord*vec2(srcw,-srch)+vec2(srcx,-srcy+0.5);
}