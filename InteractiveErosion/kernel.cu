#include "kernel.cuh"

using namespace std;

void refreshGL() {
	DWORD dwWaitResult;
	dwWaitResult = WaitForSingleObject(
		simMutex,    // handle to mutex
		100);  // no time-out interval
	if (dwWaitResult != WAIT_OBJECT_0) { return; }
	dwWaitResult = WaitForSingleObject(
		glMutex,    // handle to mutex
		100);  // no time-out interval
	if (dwWaitResult != WAIT_OBJECT_0) { ReleaseMutex(simMutex); return; }
	call::passShell(display::getShellInterlop(), display::getWallInterlop(),
		display::getMaskInterlop(), display::getTexInterlop(),
		display::getWaterInterlop(), display::getWaterMaskInterlop(),
		display::getSedimentMaskInterlop(),
		&simMap, tweakLayers::getSelectedLayer() + 1, tweakLayers::getShowAll(), display::getScale());
	ReleaseMutex(glMutex);
	ReleaseMutex(simMutex);
}

void refreshGLcb() {
	//No callback action for now.
}

//-----------------------

void keyboard(unsigned char key, int x, int y) {
	int twreturn = TwEventKeyboardGLUT(key, x, y);
	if (twreturn != 0) {
		glutPostRedisplay();
		return;
	}
	if (key == 'f') {
		printf("Toggle Fullscreen\n");
		glutFullScreenToggle();
	}
	if (key == 'v') {
		printf("Forced GL Refresh\n");
		refreshGL();
	}
	glm::mat4 vview = glm::rotate(glm::mat4(1.0f), glm::radians(-camera.pitchAngle), glm::vec3(1, 0, 0));
	glm::mat4 hview = glm::rotate(glm::mat4(1.0f), glm::radians(-camera.yawAngle), glm::vec3(0, 1, 0));
	if (key == 'a') camera.position = camera.position + cameraSpd * dedim(hview*vview*glm::vec4(1,0,0,0));
	if (key == 'd') camera.position = camera.position + cameraSpd * dedim(hview*vview*glm::vec4(-1, 0, 0, 0));
	if (key == 'q') camera.position = camera.position + cameraSpd * dedim(hview*vview*glm::vec4(0, 1, 0, 0));
	if (key == 'e') camera.position = camera.position + cameraSpd * dedim(hview*vview*glm::vec4(0, -1, 0, 0));
	if (key == 'w') camera.position = camera.position + cameraSpd *  dedim(hview*vview*glm::vec4(0, 0, 1, 0));
	if (key == 's') camera.position = camera.position + cameraSpd * dedim(hview*vview*glm::vec4(0, 0, -1, 0));
	glutPostRedisplay();
}

void mouseWheel(int button, int dir, int x, int y) {
	int ui = uiSystem.isUIOver((float)x, (float)y, false);
	if (GLUT_ACTIVE_SHIFT == glutGetModifiers()) {
		cameraSpd += dir;
		if (cameraSpd < 1) {
			cameraSpd = 1;
		}
		TwRefreshBar(bar);
		glutPostRedisplay();
	}
	else {
		//the alternative code lets the user zoom in and out with the wheel as well.
		glm::mat4 vview = glm::rotate(glm::mat4(1.0f), glm::radians(-camera.pitchAngle), glm::vec3(1, 0, 0));
		glm::mat4 hview = glm::rotate(glm::mat4(1.0f), glm::radians(-camera.yawAngle), glm::vec3(0, 1, 0));
		if (dir > 0) camera.position = camera.position + dedim(hview*vview*glm::vec4(0, 0, 10, 0));
		if (dir < 0) camera.position = camera.position + dedim(hview*vview*glm::vec4(0, 0, -10, 0));
		glutPostRedisplay();
	}
}

glm::vec3 unprojectPointer(int x, int y) {
	float h = glutGet(GLUT_WINDOW_HEIGHT);
	float w = glutGet(GLUT_WINDOW_WIDTH);
	float depth = readDepth(x, h - y);
	glm::mat4 mv = model * view;
	glm::vec3 wincoord = glm::vec3((2 * x / w) - 1, 1 - (2 * y / h), depth);
	glm::vec4 view2 = glm::vec4(-1.0f, -1.0f, 2, 2);
	glm::vec3 objcoord = glm::unProject(wincoord, mv, proj, view2);
	pointer.x = objcoord[0];
	pointer.y = objcoord[1];
	pointer.z = objcoord[2];
	return objcoord;
}

void mouseMotiontionPassive(int x, int y) {
	int twreturn = TwEventMouseMotionGLUT(x, y);
	if (twreturn != 0) {
		glutPostRedisplay();
	}
	int ui = uiSystem.isUIOver((float)x, (float)y, false);
	if (ui != -1) {
	} else {
		unprojectPointer(x,y);
		glutPostRedisplay();
	}
}

void loadTerrain() {
	int mb = 32;
	size_t cudaHeapSize = 1024 * 1000 * mb;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, cudaHeapSize);
	//
	const char * bmp_file = "terrain.bmp";
	printf("Loading Terrain...\n");
	simMap.init(terrainWidth, terrainHeight, 10);
	simMutex = CreateMutex(
		NULL,              // default security attributes
		FALSE,             // initially not owned
		"simMutex");             // named mutex
	glMutex = CreateMutex(
		NULL,              // default security attributes
		FALSE,             // initially not owned
		"glMutex");             // named mutex
	simMap.setMutex(simMutex);
	//material
	SimMaterial::ErosionParam param_rock = { 0.05, 75, 0.1, 0.1 };
	SimMaterial::ErosionParam param_soil = { 0.6, 30, 0.5, 0.5 };
	SimMaterial::ErosionParam param_dust = { 0.8, 15, 0.9, 0.9 };
	simMap.addMaterial(0, "bedrock", "tex/rock.bmp", param_rock);
	simMap.addMaterial(1, "soil", "tex/soil.bmp", param_soil);
	simMap.addMaterial(2, "dust", "tex/dust.bmp", param_dust);
	//layers
	unsigned int tmp_imageSize, tmp_height, tmp_width;
	//
	simMap.addLayer(0, "Bedrock Layer", 0);
	simMap.addLayer(1, "Soil Layer", 1);
	simMap.addLayer(2, "Dust Layer", 2);
	//
	simMap.setLayerDataFromFile("data/1.bmp", 0, true, true, true, 1);
	simMap.setLayerDataFromFile("data/2.bmp", 1, true, true, true, 1);
	simMap.setLayerDataFromFile("data/3raw.bmp", 2, true, true, true, 1);
	//Set up display
	display::genDisplayElements(terrainWidth, terrainHeight, 10, 0.5);
	//Create Ant Tweak Bar UI for our map.
	tweakLayers::genLayerPanel(&simMap);
	tweakMaterials::genMaterialPanel(&simMap);
}

void mouseButton(int button, int state, int x, int y) {
	if (state == GLUT_UP) {
		mouseDown = false;
		if (toolMode == M_POLY_GEN) {
			simMap.getPoly()->drag = false;
		}
	}
	int twreturn = TwEventMouseButtonGLUT(button, state, x, y);
	if (twreturn != 0) {
		glutPostRedisplay();
		return;
	}
	//UI detection
	int ui = uiSystem.isUIOver((float)x, (float)y, state == GLUT_DOWN);
	int ui2 = uiSystem2.isUIOver((float)x, (float)y, state == GLUT_DOWN || state == GLUT_UP);
	if (ui != -1 || ui2 != -1) {
		if (ui != -1) {
			uiSystem.deactivateOther(ui);
			toolMode = uiSystem.getButton(ui).getId();
			if (!uiSystem.getButton(ui).clicked) {
				toolMode = -1;
			}
		}
		if (ui2 != -1) {
			displayMode = ui2;
		}
		glutPostRedisplay();
	} else {//non UI
		// only start motion if the right button is pressed
		if (button == GLUT_RIGHT_BUTTON) {
			// when the button is released
			if (state == GLUT_UP) {
				cameradrag = false;
				cameramove = false;
				camera.pitchAngle += deltaAngle;
				camera.yawAngle += deltaAngle2;
				yOrigin = -1;
				xOrigin = -1;
			}
			else {// state = GLUT_DOWN
				cameradrag = true;
				cameramove = false;
				yOrigin = y;
				xOrigin = x;
			}
		}
		if (button == GLUT_MIDDLE_BUTTON) {
			// when the button is released
			if (state == GLUT_UP) {
				cameradrag = false;
				cameramove = false;
				//SALAT
				yOrigin = -1;
				xOrigin = -1;
			}
			else {// state = GLUT_DOWN
				cameradrag = true;
				cameramove = true;
				yOrigin = y;
				xOrigin = x;
			}
		}
		if (button == GLUT_LEFT_BUTTON) {
			glm::vec3 pos;
			pos = unprojectPointer(x, y);
			if (state == GLUT_UP) {
				mouseDown = false;
				if (toolMode == M_POLY_GEN) {
					simMap.getPoly()->drag = false;
				}
			}
			else {
				mouseDown = true;
				if (toolMode == M_POLY_GEN) {
					const int point = simMap.getPoly()->getPointByPosition(pos.x, pos.z,6);
					if (simMap.getPoly()->finished) {
						if (point != -1) {
							simMap.getPoly()->drag = true;
							simMap.getPoly()->dragIdx = point;
						}
					}
					else {
						if (simMap.getPoly()->getVector().size() > 2)
							if (point == 0) {
								simMap.getPoly()->addPoint(simMap.getPoly()->getVector()[0]);
								simMap.getPoly()->finished = true;
								mask::polyMask(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
								return;
							}
						simMap.getPoly()->addPoint(pos);
					}
				}
				else if (toolMode == M_POLY_DEL) {
					const int point = simMap.getPoly()->getPointByPosition(pos.x, pos.z, 6);
					if (point != -1) {
						if (point == 0 || simMap.getPoly()->getVector().size() - 1) {
							simMap.getPoly()->setPoint(0, simMap.getPoly()->getVector()[simMap.getPoly()->getVector().size() - 2]);
							simMap.getPoly()->deletePoint(simMap.getPoly()->getVector().size() - 1);
						}
						else {
							simMap.getPoly()->deletePoint(point);
						}
						if (simMap.getPoly()->getVector().size() <= 3) {
							simMap.getPoly()->resetPoly();
							simMap.getPoly()->finished = false;
							toolMode = M_POLY_GEN;
						}
						else {
							mask::polyMask(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
						}
					}
				}
				else if (toolMode == M_SPRINKLER_GEN) {
					simMap.getSprinkler()->addPoint(pos);
				}
				else if (toolMode == M_SPRINKLER_DEL) {
					const int point = simMap.getSprinkler()->getPointByPosition(pos.x, pos.z, 10);
					if (point != -1) {
						simMap.getSprinkler()->deletePoint(point);
					}
				}
			}
			mousePosX = x;
			mousePosY = y;
		}
	}
}

void toolActivity() {
	if (mouseDown) {
		glm::vec3 pos;
		switch (toolMode) {
		case M_UP:
			sculpt::sculptHeight(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		case M_DOWN:
			sculpt::sculptHeight(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, -1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		case M_THERMAL:
			thermal::erodeThermal(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, -1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		case M_HYDRAULIC:
			hydro::simWater(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), erode_k, erode_h, tweakLayers::getSelectedLayer(), toolStrength, toolRadius, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		case M_POLY_GEN:
			pos = unprojectPointer(mousePosX, mousePosY);
			if (simMap.getPoly()->drag == true) {
				if (simMap.getPoly()->dragIdx == 0) {
					simMap.getPoly()->setPoint(simMap.getPoly()->getVector().size()-1, pos);
				}
				simMap.getPoly()->setPoint(simMap.getPoly()->dragIdx, pos);
				mask::polyMask(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			}
			else if(!simMap.getPoly()->finished) {
				if (simMap.getPoly()->getVector().size() > 2) {
					simMap.getPoly()->setPoint(simMap.getPoly()->getVector().size()-1, pos);
				}
			}
			break;
		case M_PAINT_GEN:
			paint::paintMask(true, pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		case M_PAINT_DEL:
			paint::paintMask(true, pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, -1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		case M_PAINTW_GEN:
			paint::paintMask(false, pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		case M_PAINTW_DEL:
			paint::paintMask(false, pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), toolStrength, toolRadius, -1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
			break;
		}
		glutPostRedisplay();
	}
}

void mouseMotionActive(int x, int y) {
	int twreturn = TwEventMouseMotionGLUT(x, y);
	if (twreturn != 0) {
		glutPostRedisplay();
		return;
	}
	int ui = uiSystem.isUIOver((float)x, (float)y, false);
	int ui2 = uiSystem2.isUIOver((float)x, (float)y, false);
	if (ui != -1 || ui2 != -1) {
		//UI callback
	}
	else {
		if (cameradrag) {
			if (cameramove) {
				glm::mat4 vview = glm::rotate(glm::mat4(1.0f), glm::radians(-camera.pitchAngle), glm::vec3(1, 0, 0));
				glm::mat4 hview = glm::rotate(glm::mat4(1.0f), glm::radians(-camera.yawAngle), glm::vec3(0, 1, 0));
				camera.position = camera.position + (float)(x - xOrigin) * 0.35f * dedim(hview*glm::vec4(1, 0, 0, 0));
				camera.position = camera.position + (float)(y - yOrigin) * 0.35f * dedim(hview*glm::vec4(0, 0, 1, 0));
				xOrigin = x;
				yOrigin = y;
			}
			else {
				// this will only be true when the button is down
				if (yOrigin >= 0) {
					// update deltaAngle
					deltaAngle = (y - yOrigin) * 0.3f;
					camera.pitchAngle += deltaAngle;
					deltaAngle = 0;
					yOrigin = y;
				}
				if (xOrigin >= 0) {
					// update deltaAngle
					deltaAngle2 = (x - xOrigin) * 0.3f;
					camera.yawAngle += deltaAngle2;
					deltaAngle2 = 0;
					xOrigin = x;
				}
			}
		}
		mousePosX = x;
		mousePosY = y;
		pointer = unprojectPointer(x, y);
		glutPostRedisplay();
	}
}

void renderTerrainMap(void) {
	refreshGL();
	DWORD dwWaitResult;
	dwWaitResult = WaitForSingleObject(
		glMutex,    // handle to mutex
		10);  // time-out interval
	if (dwWaitResult == WAIT_OBJECT_0) {
		display::displayCall(camera, tweakLayers::getShowAll() ? simMap.getLayerCount() : tweakLayers::getSelectedLayer() + 1,
			pointer, &view, &proj, &model, toolRadius, displayMode, toolMode, &simMap, farview);
		ReleaseMutex(glMutex);
	}
}

void reshape(int width, int height) {
	TwWindowSize(width, height);
	glViewport(0, 0, width, height);
	refreshTool();
}

void displayCall(void) {
	renderTerrainMap();
	uiSystem.render();
	uiSystem2.render();
	TwDraw();
	glutSwapBuffers();
}

void updateScene(void) {
	//check exit request
	if (requestExit) {
		glutLeaveMainLoop();
		return;
	}
	//calling of individual tools
	toolActivity();
	//pass all buffered changes to the polygon
	simMap.getPoly()->shiftPoints();
	simMap.getSprinkler()->shiftPoints();
	//simulation update
	if (autoupdate) {
		if (updateTimer == NULL) {
			updateTimer = clock();
		}
		else {
			clock_t now = clock();
			clock_t passed = now - updateTimer;
			float secs = (float)passed / CLOCKS_PER_SEC;
			if (secs < updateInterval/1000.0f) {
				return;
			}
			else {
				updateTimer = clock();
			}
		}
		pass::passData(&simMap,simMutex);
		hydro::simWater(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), erode_k, erode_h, tweakLayers::getSelectedLayer(), 0, 0, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
		if(erode_t)
			thermal::erodeThermal(pointer.x / display::getScale(), pointer.y / display::getScale(), pointer.z / display::getScale(), tweakLayers::getSelectedLayer(), 0, 0, 1, &simMap, simMutex, &refreshGLcb, sprinklerStrength, sprinklerRadius, evaporation);
		glutPostRedisplay();
	}
}

void refreshTool() {
	TwDefine(" GLOBAL help='This application provides some basic tool for terrain erosion.' "); // Message added to the help bar.
	string param = " Tool label='Tool Options' position='" + std::to_string(glutGet(GLUT_WINDOW_WIDTH) - (std::max(170, (glutGet(GLUT_WINDOW_HEIGHT) / 4))) - 15) +
		" " + std::to_string(12 * glutGet(GLUT_WINDOW_HEIGHT) / 20) + "' size='" + std::to_string(std::max(170, (glutGet(GLUT_WINDOW_HEIGHT) / 4))) + " " +
		std::to_string(8 * glutGet(GLUT_WINDOW_HEIGHT) / 20 - 30) +
		"' valueswidth='" + std::to_string(glutGet(GLUT_WINDOW_HEIGHT) / 20) +
		"' iconifiable='false' movable='false' resizable='false' color='96 216 224' ";
	TwDefine(param.c_str());
}

void initGlut(int argc, char ** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	glutInitWindowPosition(50, 50);
	glutInitWindowSize(W, H);
	glutCreateWindow("Interactive Erosion");
	glutCreateMenu(NULL);
	//
	GLenum errnum = glewInit();
	if (errnum != GLEW_OK) {
		printf("glewInit failed %d\n", errnum);
	}
	//
	glutDisplayFunc(displayCall);
	//
	// Initialize AntTweakBar
	TwInit(TW_OPENGL, NULL);
	TwWindowSize(W, H);
	glutSpecialFunc((GLUTspecialfun)TwEventSpecialGLUT);
	TwGLUTModifiersFunc(glutGetModifiers);
	//
	// Create a tweak bar
	bar = TwNewBar("Tool");
	refreshTool();

	TwAddVarRW(bar, "Radius", TW_TYPE_FLOAT, &toolRadius,
		" min=1 max=500 step=5 keyIncr=r keyDecr=R help='Radius of the editing tool.' ");
	TwAddVarRW(bar, "Strength", TW_TYPE_FLOAT, &toolStrength,
		" min=0 max=1000 step=10 keyIncr=t keyDecr=T help='Strength of the editing tool.' ");
	TwAddVarRW(bar, "Camera Speed", TW_TYPE_FLOAT, &cameraSpd,
		" min=1 max=150 step=1 help='Speed with which you move the camera.' ");
	TwAddVarRW(bar, "Camera Draw Distance", TW_TYPE_FLOAT, &farview,
		" min=100 step=25 help='How far the camera renders.' ");
	TwAddVarRW(bar, "Evaporation", TW_TYPE_FLOAT, &evaporation,
		" min=0 step=1 help='Speed with which water evaporates.' ");
	TwAddVarRW(bar, "Sprinkler Strength", TW_TYPE_FLOAT, &sprinklerStrength,
		" min=0 step=0.01 help='Amount of water generated by sprinklers.' ");
	TwAddVarRW(bar, "Sprinkler Radius", TW_TYPE_FLOAT, &sprinklerRadius,
		" min=0 step=1 help='Radius of the sprinklers.' ");
	TwAddSeparator(bar, "Erosion Commands", NULL);
	TwAddVarRW(bar, "Thermal", TW_TYPE_BOOLCPP, &erode_t,
		" help='Conducts thermal erosion during update loop.' ");
	TwAddVarRW(bar, "Kinetic", TW_TYPE_BOOLCPP, &erode_k,
		" help='Conducts kintetic water flow erosion during update loop.' ");
	TwAddVarRW(bar, "Hydraulic", TW_TYPE_BOOLCPP, &erode_h,
		" help='Conducts hydraulic water dissolution erosion during update loop.' ");
	TwAddVarRW(bar, "Sim Interval (ms)", TW_TYPE_FLOAT, &updateInterval,
		" min=1 max=10000 step=100 help='Interval between auto simulation steps in milliseconds.' ");
	TwAddVarRW(bar, "> TSimulate <", TW_TYPE_BOOLCPP, &autoupdate,
		" keyIncr=Space keyDecr=Space help='When active, the application will periodically erode the terrain.' ");
	//
	glutIdleFunc(updateScene);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseWheelFunc(mouseWheel);
	glutMotionFunc(mouseMotionActive);
	glutPassiveMotionFunc(mouseMotiontionPassive);
	glutMouseFunc(mouseButton);
}

void genUI(GLuint progID) {
	uiSystem = UISystem(3.0 / 4.0);
	uiSystem2 = UISystem(3.0 / 4.0);
	uiSystem.genArrays(progID);
	uiSystem2.genArrays(progID);
	float offset = 0.11;
	float b1x = -0.23; 
	float b1y = -0.38;
	float b2x = 0.02;
	float b2y = -0.38;
	UIButton b1 = UIButton(M_UP, UISprite(Right, loadBMP_custom("ui/uiup.bmp"), b1x, b1y + 0 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1));
	b1.clicked = true;
	uiSystem.addButton(b1);
	uiSystem.addButton(UIButton(M_DOWN, UISprite(Right, loadBMP_custom("ui/uidown.bmp"), b1x + offset, b1y + 0*offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_POLY_DEL, UISprite(Right, loadBMP_custom("ui/uipolym.bmp"), b1x + offset, b1y + 1 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_POLY_GEN, UISprite(Right, loadBMP_custom("ui/uipolyp.bmp"), b1x, b1y + 1 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_SPRINKLER_DEL, UISprite(Right, loadBMP_custom("ui/uisprm.bmp"), b1x + offset, b1y + 4 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_SPRINKLER_GEN, UISprite(Right, loadBMP_custom("ui/uisprp.bmp"), b1x, b1y + 4 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_PAINT_GEN, UISprite(Right, loadBMP_custom("ui/uipaintp.bmp"), b1x, b1y + 2 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_PAINT_DEL, UISprite(Right, loadBMP_custom("ui/uipaintm.bmp"), b1x + offset, b1y + 2 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_PAINTW_GEN, UISprite(Right, loadBMP_custom("ui/uiwaterp.bmp"), b1x, b1y + 3 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem.addButton(UIButton(M_PAINTW_DEL, UISprite(Right, loadBMP_custom("ui/uiwaterm.bmp"), b1x + offset, b1y + 3 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	///=================
	uiSystem2.addButton(UIButton(VREN_COMBINED, UISprite(Left, loadBMP_custom("ui/uiviewmix.bmp"), b2x, b2y + offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem2.addButton(UIButton(VREN_MASK, UISprite(Left, loadBMP_custom("ui/uiviewsel.bmp"), b2x, b2y + 2 * offset, 0.1, 0.1, 0, 0.5, 0.5, 1)));
	uiSystem2.addButton(UIButton(VREN_TEXTURE, UISprite(Left, loadBMP_custom("ui/uiviewtex.bmp"), b2x, b2y, 0.1, 0.1, 0, 0.5, 0.5, 1)));
}

void readDir(char* dirName) {
	struct stat s;
	if (stat(dirName, &s) == 0)
	{
		if (s.st_mode & S_IFDIR)
		{
			//it's a directory
			printf("dir");
		}
		else if (s.st_mode & S_IFREG)
		{
			//it's a file
			printf("file");
		}
		else
		{
			//something else
			printf("else");
		}
	}
	else
	{string
		//error
		printf("err");
	}
	//fclose(dataDir);
	cout << "\nread\n";
}

void initThreads() {
	thermal::initThread();
	hydro::initThread();
	sculpt::initThread();
	mask::initThread();
	paint::initThread();
	pass::initThread();
}

void killThreads() {
	thermal::killThread();
	hydro::killThread();
	sculpt::killThread();
	mask::killThread();
	paint::killThread();
	pass::killThread();
}

int main(int argc, char ** argv) {
	printf("Welcome to Interactive Erosion \n\n");
	printf("created by David Hrusa\nver. 1.0\n\n--------------------\nThis application requires a CUDA compatible GPU to operate.\n\n");
	terrainHeight = 256;
	terrainWidth = 256;
	terrainScale = 0.2;
	//
	initGlut(argc, argv);
	display::loadShaders();
	loadTerrain();
	tweakInit::genInitWizard(simMutex, glMutex, &simMap, &display::genDisplayElements);
	initThreads();
	genUI(display::getUIShader());
	//Main Loop
	printf(">>> Application Initialized <<<\n");
	glutMainLoop();
	//Clean up threads and memory after execution
	printf(">>> CLOSING <<<\n");
	DWORD dwWaitResult;
	killThreads();
	dwWaitResult = WaitForSingleObject(
		simMutex,    // handle to mutex
		INFINITY);  // no time-out interval
	return 0;
}