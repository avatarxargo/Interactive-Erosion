#include "tweakMaterials.h"
#define MAXMATERIALS 19

namespace tweakMaterials {

	TwBar *twBar;
	unsigned int selectedMaterial = 0;
	bool mat[20];
	bool locked = false;
	bool up = false;
	bool showall = true;
	int indexlist[20] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 };
	unsigned int activeMaterialCount = 3;
	unsigned int viscount = 1;
	SimMap* simMap;

	void refreshLayers();

	static bool RETVALSL[2] = { true, false };

	void unselectAll() {
		for (int i = 0; i < activeMaterialCount; ++i) {
			mat[i] = false;
		}
	}

	void TW_CALL setLayerCallback(const void *value, void *clientData)
	{
		if (*(const bool *)value == false || locked) { return; }
		selectedMaterial = *(int *)clientData;
		unselectAll();
		mat[selectedMaterial] = *(const bool *)value;
	}

	void TW_CALL getLayerCallback(void *value, void *clientData)
	{
		*(bool *)value = mat[*(int *)clientData];
	}

	void cb_add_accept(char * file) {
		activeMaterialCount += 1;
		unselectAll();
		if (up) {
			selectedMaterial++;
		}
		int passidx = (int)selectedMaterial;
		printf("Adding layer to: %d, %d\n", passidx, selectedMaterial);
		simMap->callbackAddMaterial(file, passidx);
		refreshLayers();
		mat[selectedMaterial] = true;
		locked = false;
	}

	void cb_add_cancel() {
		locked = false;
	}

	//callback wrapper for adding a layer
	void TW_CALL cb_add(void *clientData) {
		if (locked) { return; }
		if (activeMaterialCount < MAXMATERIALS) {
			locked = true;
			up = *(bool*)clientData;
			tweakSimpleWizard::genLayerWizard("tex", cb_add_cancel, cb_add_accept, up);
		}
	}

	//callback wrapper for moving a layer
	void TW_CALL cb_move(void *clientData) {
		if (locked) { return; }
		up = *(bool*)clientData;
		if (up) {
			if (selectedMaterial >= activeMaterialCount) { return; }
		}
		else {
			if (selectedMaterial == 0) { return; }
		}
		simMap->callbackMove(selectedMaterial, up);
		if (up) {
			selectedMaterial++;
		}
		else {
			selectedMaterial--;
		}
		unselectAll();
		refreshLayers();
		mat[selectedMaterial] = true;
	}

	//callback wrapper for moving a layer
	void TW_CALL cb_remove(void *clientData) {
		if (locked || selectedMaterial==0) { return; }
		if (activeMaterialCount > 1) {
			simMap->callbackRemoveMaterial(selectedMaterial);
			activeMaterialCount -= 1;
			if (selectedMaterial >= activeMaterialCount) {
				unselectAll();
				selectedMaterial = activeMaterialCount - 1;
				mat[selectedMaterial] = true;
			}
			refreshLayers();
		}
	}

	//updates the list of
	void refreshLayers() {
		TwAddVarCB(twBar, "tmp", TW_TYPE_BOOLCPP, &setLayerCallback, &getLayerCallback, &indexlist[0], "group='Material List'");
		std::string str = "Material";
		for (int i = 1; i <= viscount; ++i) {
			str = "Material" + std::to_string(i);
			const char *cstr = str.c_str();
			TwRemoveVar(twBar, cstr);
		}
		for (int i = activeMaterialCount - 1; i >= 0; --i) {
			str = "Material" + std::to_string(i + 1);
			std::string strparam = "group = 'Material List' label = '" + simMap->getMaterial(i)->name + "'";
			const char *cstr = str.c_str();
			TwAddVarCB(twBar, cstr, TW_TYPE_BOOLCPP, &setLayerCallback, &getLayerCallback, &indexlist[i], strparam.c_str());
		}
		viscount = activeMaterialCount;
		TwRemoveVar(twBar, "tmp");
	}

	void TW_CALL SetNameStr(const void *value, void *clientData)
	{
		const std::string *srcPtr = static_cast<const std::string *>(value);
		simMap->getMaterial(selectedMaterial)->name = *srcPtr;
		std::string str = " med/Material" + std::to_string(selectedMaterial+1) + " label='" + srcPtr->c_str() + "' ";
		TwDefine(str.c_str());
	}
	void TW_CALL GetNameStr(void *value, void *clientData)
	{
		std::string *destPtr = static_cast<std::string *>(value);
		TwCopyStdStringToLibrary(*destPtr, simMap->getMaterial(selectedMaterial)->getName());
	}

	const int FLOATVAR[4] = { 0,1,2,3 };

	void TW_CALL SetFloat(const void *value, void *clientData)
	{
		float *srcPtr = (float *)(value);
		switch (*(int*)clientData) {
		case 0:
			simMap->getMaterial(selectedMaterial)->getParamsPtr()->thermalRate = *srcPtr;
			break;
		case 1:
			simMap->getMaterial(selectedMaterial)->getParamsPtr()->talosAngle = *srcPtr;
			break;
		case 2:
			simMap->getMaterial(selectedMaterial)->getParamsPtr()->hydroRate = *srcPtr;
			break;
		case 3:
			simMap->getMaterial(selectedMaterial)->getParamsPtr()->sedimentRate = *srcPtr;
			break;
		}
	}
	void TW_CALL GetFloat(void *value, void *clientData)
	{
		float *destPtr = static_cast<float *>(value);
		switch (*(int*)clientData) {
		case 0:
			*destPtr = simMap->getMaterial(selectedMaterial)->getParams().thermalRate;
			break;
		case 1:
			*destPtr = simMap->getMaterial(selectedMaterial)->getParams().talosAngle;
			break;
		case 2:
			*destPtr = simMap->getMaterial(selectedMaterial)->getParams().hydroRate;
			break;
		case 3:
			*destPtr = simMap->getMaterial(selectedMaterial)->getParams().sedimentRate;
			break;
		}
	}

	void genMaterialPanel(SimMap* argsimMap) {
		simMap = argsimMap;
		twBar = TwNewBar("med");
		TwDefine(" med size='300 300' color='196 205 116 124' position='70 400' label='Material Editor'"); // change default tweak bar size and color
																	   // Add 'g_Zoom' to 'bar': this is a modifable (RW) variable of type TYPE_FLOAT. Its key shortcuts are [z] and [Z].
																	   //float g_Zoom = 1.0f;
		TwAddVarRO(twBar, "Selected Material", TW_TYPE_UINT32, &selectedMaterial,
			" min=0 max=2 step=1 keyIncr=l keyDecr=L help='Currently selected material for editing operations' ");
		TwAddSeparator(twBar, NULL, NULL);
		mat[0] = true;
		TwAddVarCB(twBar, "Material1", TW_TYPE_BOOLCPP, &setLayerCallback, &getLayerCallback, &indexlist[0], "group='Material List'");
		TwAddSeparator(twBar, NULL, NULL);
		TwAddVarCB(twBar, "stat0ed", TW_TYPE_STDSTRING, &SetNameStr, &GetNameStr, NULL, " label='Material Name:' group='Material Properties' ");
		TwAddVarCB(twBar, "stat1ed", TW_TYPE_FLOAT, &SetFloat, &GetFloat, (void*)&FLOATVAR[0], " label='Thermal Erosion Rate:' group='Material Properties' help='Rate at which material crumbles when layered too steeply.' ");
		TwAddVarCB(twBar, "stat2ed", TW_TYPE_FLOAT, &SetFloat, &GetFloat, (void*)&FLOATVAR[1], " label='Stable Slope Angle:' group='Material Properties' help='An angle under which the material will no longer crumble.' ");
		TwAddVarCB(twBar, "stat3ed", TW_TYPE_FLOAT, &SetFloat, &GetFloat, (void*)&FLOATVAR[2], " label='Hydraulic Erosion Rate:' group='Material Properties' help='How quickly this material dissolves in water.' ");
		TwAddVarCB(twBar, "stat4ed", TW_TYPE_FLOAT, &SetFloat, &GetFloat, (void*)&FLOATVAR[3], " label='Sedimentation Rate:' group='Material Properties' help='How quickly this material settles.' ");
		TwAddButton(twBar, "addmat", cb_add, &RETVALSL[0], " label='Add Material' group='Material Actions' ");
		TwAddButton(twBar, "removemat", cb_remove, &RETVALSL[1], " label='Remove Material' group='Material Actions' ");
		refreshLayers();
	}

	int getSelectedMaterial() {
		return selectedMaterial;
	}

	bool getShowAll() {
		return showall;
	}

}