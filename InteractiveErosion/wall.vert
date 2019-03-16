#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in float mask;
layout(location = 2) in float texmask;

// Position for coloring the mesh 
out vec3 pos;

// Texture coordinate
out vec2 Texcoord;

// Mask
out float maskVert; 

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main(){
	maskVert = mask;
    // Output position of the vertex, in clip space : MVP * position
    gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
	
    pos = vertexPosition_modelspace.xyz;
	Texcoord = vec2((vertexPosition_modelspace.x+vertexPosition_modelspace.z)*0.03/0.4,mask);//vertexPosition_modelspace.xz*0.03/0.4;
}