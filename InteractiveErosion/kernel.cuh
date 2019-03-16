#pragma once
#include <CImg.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <windows.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include "loadbmp.h"
#include "camera3D.h"
#include "terrainvar.h"
#include "tools.h"
#include "display.h"
#include "globals.h"
#include "call.cuh"
#include "mask.cuh"
#include "sculpt.cuh"
#include "paint.cuh"
#include "pass.cuh"
#include "thermal.cuh"
#include "hydro.cuh"
#include "tweakLayers.h"
#include "tweakMaterials.h"
#include "tweakInit.h"
#include <mutex>
#include <glm/glm.hpp>

//number indexing for various modes the user may employ
//unused
#define M_CAMERA 0
//paint hills
#define M_UP 1
#define M_DOWN 2
//polygon mask
#define M_POLY_GEN 3
#define M_POLY_DEL 9
//paint mask
#define M_PAINT_GEN 4
#define M_PAINT_DEL 8
//erosion types
#define M_THERMAL 5
#define M_HYDRAULIC 6
//water generating sprinklers
#define M_SPRINKLER_GEN 7
#define M_SPRINKLER_DEL 10
//painting water manualy
#define M_PAINTW_GEN 11
#define M_PAINTW_DEL 12
int toolMode = 1;

//view modes
#define VREN_TEXTURE 2
#define VREN_MASK 0
#define VREN_COMBINED 1

int displayMode = 0;

//mutexes for thread control.
HANDLE simMutex, glMutex;

glm::vec3 pointer;
float toolRadius = 50;
float toolStrength = 150;
float cameraSpd = 1;
float evaporation = 1;
float sprinklerStrength = 0.1;
float sprinklerRadius = 10;
float farview = 300;
bool autoupdate = false;
bool cameradrag = false;
bool cameramove = false;
bool mouseDown = false;
bool shiftPressed = false;
//
bool erode_t = true;
bool erode_k = true;
bool erode_h = true;
//
bool requestExit = false;

clock_t updateTimer;
float updateInterval = 1;

// Pointer to the tool tweak bar
TwBar *bar;

//Camera controls
float deltaAngle = 0.0f;
float deltaAngle2 = 0.0f;
int xOrigin = -1;
int yOrigin = -1;
int mouseX = 0;
int mouseY = 0;
//on screen mouse coords
float mousePosX = 0;
float mousePosY = 0;
//

//Terrain
UISystem uiSystem;
UISystem uiSystem2;
Camera3D camera;
SimMap simMap;

glm::mat4 mvp;
glm::mat4 proj;
glm::mat4 view;
glm::mat4 model;
//

void refreshTool();
void toolActivity();
void killThreads();
void initThreads();

#pragma comment(lib, "freeglut.lib")
#pragma comment(lib, "glew32.lib")