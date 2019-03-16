#version 330 core

in vec3 pos;

// Texture coordinate
in vec2 Texcoord;

// Mask
in float maskVert; 

// Ouput data
out vec4 color;

// Size of the mesh for scaling color
uniform vec3 meshsize;
uniform vec3 pointer;
uniform vec3 polygon[100];
uniform int polygonLength;
uniform int renderMode;
uniform int connectPoly;
uniform int showGauss;
uniform float toolRadius;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D texes[10];

void main(){
	float opacity = 1;
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
		color = vec4(1,1,1,opacity);
	} else if (mod(Texcoord.x,1)<0.02 || mod(Texcoord.y,1)<0.03 || close) {
		color = vec4(0.2+max(0.8-maskVert*0.8,0),0.2+(maskVert*0.8),1,opacity);
	} else if (mod(Texcoord.x*3,1)<0.01 || mod(Texcoord.y*3,1)<0.015) {
		color = vec4(0.2+max(0.8-maskVert*0.8,0),0.2+(maskVert*0.8),1,opacity);
	} else if (mod(Texcoord.x*6,1)<0.01 || mod(Texcoord.y*6,1)<0.02) {
		color = vec4(0.2+max(0.8-maskVert*0.8,0),0.2+(maskVert*0.8),1,opacity);
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
			if(maskVert<1) {
				color = texture(texes[0], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<2) {
				color = texture(texes[1], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<3) {
				color = texture(texes[2], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<4) {
				color = texture(texes[3], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<5) {
				color = texture(texes[4], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<6) {
				color = texture(texes[5], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<7) {
				color = texture(texes[6], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<8) {
				color = texture(texes[7], vec2(Texcoord.x,-Texcoord.y));
			} else if(maskVert<9) {
				color = texture(texes[8], vec2(Texcoord.x,-Texcoord.y));
			} else {
				color = vec4(1,0,0,0.5);
			}
			return;
		}
		if(renderMode == 4) {
			color = vec4(1.0,1.0,1.0,1.0);
		}
	}
}