#version 330 core

in vec3 pos;

// Texture coordinate
in vec2 Texcoord;

// Mask
in float maskVert; 
in float texmaskVert; 

// Ouput data
out vec4 color;

// Size of the mesh for scaling color
uniform vec3 meshsize;

// Position of user pointer
uniform vec3 pointer;
uniform int showGauss;
uniform float toolRadius;

// Style of display
uniform int renderMode;

// Polygons and sprinkler entities
uniform vec3 polygon[100];
uniform vec3 sprinkler[100];
uniform int polygonLength;
uniform int sprinklerLength;
uniform int connectPoly;

// Textures
uniform sampler2D tex1;
uniform sampler2D tex2;

void main(){

	//Grabbing points for sprinklers
	for(int i = 0; i < sprinklerLength; ++i) {
		float diffx = sprinkler[i].x-pos.x+0.5;
		float diffz = sprinkler[i].z-pos.z+0.5;
		if(diffx > 0 && diffx < 1 && diffz > 0 && diffz < 1) {
			color = vec4(0.5,0.8,1,1);
			return;
		}
	}

	float opacity = min(maskVert*10,0.5);
	float sediment = texmaskVert/10;
	color = vec4(0.4+sin(Texcoord.y)/10+sediment,0.4+sin(Texcoord.x)/10+sediment,0.6,opacity+sediment);// vec4(1,1,1,0.5)*texture(tex1, vec2(Texcoord.x,-Texcoord.y));//
	return;
	
	bool close = false;
	float dst = sqrt((pointer.x - pos.x)*(pointer.x - pos.x) + (pointer.z - pos.z)*(pointer.z - pos.z));
	float distance = showGauss==1?exp(-((((pointer.x - pos.x)*(pointer.x - pos.x) + (pointer.z - pos.z)*(pointer.z - pos.z))*4) / (toolRadius*toolRadius))):0;
	for(int i=1; i<polygonLength+connectPoly-1;++i) {
		float distAB = ( (polygon[i].x-polygon[i-1].x)*(polygon[i].x-polygon[i-1].x)+(polygon[i].z-polygon[i-1].z)*(polygon[i].z-polygon[i-1].z) );
		float linedist = abs((polygon[i].z-polygon[i-1].z)*pos.x-(polygon[i].x-polygon[i-1].x)*pos.z+polygon[i].x*polygon[i-1].z-polygon[i].z*polygon[i-1].x)/sqrt(distAB);
		float distA = ( (pos.x-polygon[i].x)*(pos.x-polygon[i].x)+(pos.z-polygon[i].z)*(pos.z-polygon[i].z) );
		float distB = ( (pos.x-polygon[i-1].x)*(pos.x-polygon[i-1].x)+(pos.z-polygon[i-1].z)*(pos.z-polygon[i-1].z) );
		if(linedist<0.1 && distA<distAB && distB < distAB) {
			close = true;
		}
	}
	if(dst>toolRadius - 0.2 && dst < toolRadius) {
		close = true;
	}
	if(close) {
		color = vec4(1,1,1,1);
	} else if (mod(Texcoord.x,1)<0.02 || mod(Texcoord.y,1)<0.03 || close) {
		color = vec4(0.4+max(0.8-maskVert*0.8,0),0.4+(maskVert*0.8),1,1);
	} else if (mod(Texcoord.x*3,1)<0.01 || mod(Texcoord.y*3,1)<0.015) {
		color = vec4(0.4+max(0.8-maskVert*0.8,0),0.4+(maskVert*0.8),1,1);
	} else if (mod(Texcoord.x*6,1)<0.01 || mod(Texcoord.y*6,1)<0.02) {
		color = vec4(0.4+max(0.8-maskVert*0.8,0),0.4+(maskVert*0.8),1,1);
	} else {
		if(renderMode == 0) {
			if (mod(pos.y/3,1)<0.5) {
				color = vec4(0.8*distance+(1-maskVert)*0.8f*(pos.y+3)/10,   0.3f*log(pos.y+3)-distance/3,  (maskVert)*0.2f,opacity);
			} else {
				color = vec4(0.8*distance+(1-maskVert)*0.8f*(pos.y+1)/10,  0.3f*log(pos.y+1)-distance/3,   (maskVert)*0.2f,opacity);
			}
			return;
		}
		if(renderMode == 1) {
			color = vec4(1-maskVert,maskVert,distance,opacity);
		}
		if(renderMode == 2) {
			if (mod(pos.y/3,1)<0.5) {
				color = texture(tex1, vec2(Texcoord.x,-Texcoord.y)) * vec4(0.8*distance+0.8f*(pos.y+3)/10,   0.3f*log(pos.y+3)-distance/3,  0.2f,opacity);
			} else {
				color = texture(tex1, vec2(Texcoord.x,-Texcoord.y)) * vec4(0.8*distance+0.8f*(pos.y+1)/10,  0.3f*log(pos.y+1)-distance/3,   0.2f,opacity);
			}
			return;
		}
		if(renderMode == 4) {
			color = vec4(1.0,1.0,1.0,1);
		}
	}
}