#include "tweakInit.h"

namespace tweakInit {
	TwBar * bar;
	SimMap * simMap;
	void (*genDisplayElements)(unsigned int width, unsigned int height, unsigned int depth, float scale);
	HANDLE simHandle, glHandle;
	int width = 256, height = 256;

	void TW_CALL tw_hide(void *clientData)
	{
		TwDeleteBar(bar);
	}

	void TW_CALL tw_callback(void *clientData)
	{
		//lockMutex
		DWORD dwWaitResult;
		dwWaitResult = WaitForSingleObject(
			simHandle,    // handle to mutex
			INFINITE);
		dwWaitResult = WaitForSingleObject(
			glHandle,    // handle to mutex
			INFINITE);
		simMap->cleanseExistingLayers();
		simMap->init(width,height,10);
		//material
		SimMaterial::ErosionParam param_rock = { 0.05, 75, 0.1, 0.1 };
		SimMaterial::ErosionParam param_soil = { 0.6, 30, 0.5, 0.5 };
		SimMaterial::ErosionParam param_dust = { 0.8, 15, 0.9, 0.9 };
		if(simMap->getMaterialCount()<1)
			simMap->addMaterial(0, "bedrock", "tex/rock.bmp", param_rock);
		if (simMap->getMaterialCount()<2)
			simMap->addMaterial(1, "soil", "tex/soil.bmp", param_soil);
		if (simMap->getMaterialCount()<3)
			simMap->addMaterial(2, "dust", "tex/dust.bmp", param_dust);
		//layers
		simMap->addLayer(0, "Bedrock Layer", 0);
		simMap->addLayer(1, "Soil Layer", 1);
		simMap->addLayer(2, "Dust Layer", 2);
		simMap->getLayer(0)->blankData(10);
		simMap->getLayer(1)->blankData(5);
		simMap->getLayer(2)->blankData(1);
		simMap->passMaterialListPointers();
		simMap->passLayerListPointers();
		//release lock
		genDisplayElements(width, height, 10, 0.5);
		ReleaseMutex(simHandle);
		ReleaseMutex(glHandle);
		TwDeleteBar(bar);
	}

	void genInitWizard(HANDLE handleSim, HANDLE handleGL, SimMap * target, void (*displayFunc)(unsigned int width, unsigned int height, unsigned int depth, float scale)) {
		simMap = target;
		simHandle = handleSim;
		glHandle = handleGL;
		genDisplayElements = displayFunc;
		bar = TwNewBar("initer");
		std::string title = " initer label='Create World";
		title += "' size='500 350' position='425 250' color='205 96 116 124' ";
		TwDefine(title.c_str());
		TwAddButton(bar, "message1", NULL, NULL, " label='Welcome to Interactive Erosion.'");
		TwAddButton(bar, "message2", NULL, NULL, " label='Use the AWSDQE keys or the middle mouse button to move the camera.'");
		TwAddButton(bar, "message2b", NULL, NULL, " label='The mouse wheel to zoom. Shift+wheel to change camera speed.'");
		TwAddButton(bar, "message3", NULL, NULL, " label='The right mouse button lets you look around.'");
		TwAddButton(bar, "message4", NULL, NULL, " label='R and shift + R change the brush size.'");
		TwAddButton(bar, "message5", NULL, NULL, " label='T and shift + T change the tool strength.'");
		TwAddButton(bar, "message5b", NULL, NULL, " label='F to toggle fullscreen.'");
		TwAddButton(bar, "message6", NULL, NULL, " label='The terrain consists of layers of different materials.'");
		TwAddButton(bar, "message7", NULL, NULL, " label='To tweak the properties change the material definition.'");
		TwAddButton(bar, "message9", NULL, NULL, " label='To use custom material textures use /tex/'");
		TwAddButton(bar, "message10", NULL, NULL, " label='To use custom layer maps use /data/'");
		TwAddButton(bar, "message10b", NULL, NULL, " label='Help dialog is in the bottom left corner.'");
		TwAddButton(bar, "message12", NULL, NULL, " label=' '");
		TwAddVarRW(bar, "width", TW_TYPE_INT32, &width, "label='Width'");
		TwAddVarRW(bar, "height", TW_TYPE_INT32, &height, "label='Height'");
		TwAddSeparator(bar, NULL, NULL);
		TwAddButton(bar, "Generate New", tw_callback, NULL, NULL);
		TwAddButton(bar, "Hide This", tw_hide, NULL, NULL);
	}
}