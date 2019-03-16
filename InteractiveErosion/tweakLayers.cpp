#include "tweakLayers.h" 
#define MAXLAYERS 19

namespace tweakLayers {

	TwBar *twBar;
	unsigned int selectedLayer = 0;
	bool lay[20];
	bool locked = false;
	bool up = false;
	bool showall = true;
	int indexlist[20] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 };
	unsigned int activeLayerCount = 3;
	unsigned int viscount = 1;
	SimMap* simMap;

	void refreshLayers();

	static bool RETVALSL[2] = { true, false };

	void unselectAll() {
		for (int i = 0; i < activeLayerCount; ++i) {
			lay[i] = false;
		}
	}

	void refreshLayerMaterial() {
		std::string str = " LayerList/stat1 label='Material Name: " + simMap->getMaterial(simMap->getLayer(selectedLayer)->getMaterialIdx())->getName() + "'";
		TwDefine(str.c_str());
	}

	void TW_CALL setLayerCallback(const void *value, void *clientData)
	{
		if (*(const bool *)value == false || locked) { return; }
		selectedLayer = *(int *)clientData;
		unselectAll();
		lay[selectedLayer] = *(const bool *)value;
		refreshLayerMaterial();
	}

	void TW_CALL getLayerCallback(void *value, void *clientData)
	{
		*(bool *)value = lay[*(int *)clientData];
	}


	void TW_CALL SetMaterial(const void *value, void *clientData)
	{
		int *srcPtr = (int *)(value);
		if (*srcPtr >= simMap->getMaterialCount() || *srcPtr < 0) {
			printf("Int value %d not matched to %d \n", *srcPtr, simMap->getMaterialCount());
			return;
		}
		printf("Int value %d \n",*srcPtr);
		simMap->getLayer(selectedLayer)->setMaterialIdx(*srcPtr);
		//
		refreshLayerMaterial();
	}
	void TW_CALL GetMaterial(void *value, void *clientData)
	{
		int *destPtr = static_cast<int *>(value);
		*destPtr = simMap->getLayer(selectedLayer)->getMaterialIdx();
	}

	void cb_add_accept(char * file, int rgb, float scl) {
		activeLayerCount += 1;
		unselectAll();
		if (up) {
			selectedLayer++;
		}
		int passidx = (int)selectedLayer;
		simMap->callbackAdd(file, passidx, rgb, scl);
		refreshLayers();
		lay[selectedLayer] = true;
		locked = false;
	}

	void cb_add_cancel() {
		locked = false;
	}
	
	//callback wrapper for saveing a layer into a file with the same name.
	void TW_CALL cb_add(void *clientData) {
		if (locked) { return; }
		if (activeLayerCount < MAXLAYERS) {
			locked = true;
			up = *(bool*)clientData;
			tweakWizard::genLayerWizard("data", cb_add_cancel, cb_add_accept, up);
		}
	}

	//callback wrapper for adding a layer
	void TW_CALL cb_save(void *clientData) {
		int lw = simMap->getWidth();
		int lh = simMap->getHeight();
		std::string name = simMap->getLayer(selectedLayer)->name;
		std::string path = "data/" + name + ".bmp";
		float * d_dataptr = simMap->getLayer(selectedLayer)->getDataPtr();
		float *h_data = new float[lw*lh];
		cudaMemcpy(h_data, d_dataptr, lw*lh*sizeof(float), cudaMemcpyDeviceToHost);
		float max = 0;
		for (unsigned int i = 0; i < lw*lh; ++i) {
			if (max < h_data[i]) {
				max = h_data[i];
			}
		}
		for (unsigned int i = 0; i < lw*lh; ++i) {
			h_data[i] = h_data[i] * 255.0f / max;
		}
		cimg_library::CImg<float> img(lw,lh,1,3,0);
		for (unsigned int i = 0; i < lh; ++i) {
			for (unsigned int j = 0; j < lw; ++j) {
				img[i*lw + j] = h_data[i*lw + j];
				img[lw*lh+i*lw + j] = h_data[i*lw + j];
				img[2*lw*lh + i*lw + j] = h_data[i*lw + j];
			}
		}
		printf("Saving layer %s to file %s.\n", name.c_str(), path.c_str());
		img.save(path.c_str());
		free(h_data);
	}

	//callback wrapper for adding a blank layer
	void TW_CALL cb_addb(void *clientData) {
		if (locked) { return; }
		if (activeLayerCount < MAXLAYERS) {
			locked = true;
			up = *(bool*)clientData;
			activeLayerCount += 1;
			unselectAll();
			if (up) {
				selectedLayer++;
			}
			int passidx = (int)selectedLayer;
			simMap->callbackAddBlank(passidx);
			refreshLayers();
			lay[selectedLayer] = true;
			locked = false;
		}
	}

	//callback wrapper for moving a layer
	void TW_CALL cb_move(void *clientData) {
		if (locked) { return; }
		up = *(bool*)clientData;
		if (up) {
			if (selectedLayer+1 >= activeLayerCount) { return; }
		}
		else {
			if (selectedLayer == 0) { return; }
		}
		simMap->callbackMove(selectedLayer, up);
		if (up) {
			selectedLayer++;
		}
		else {
			selectedLayer--;
		}
		unselectAll();
		refreshLayers();
		lay[selectedLayer] = true;
	}

	//callback wrapper for moving a layer
	void TW_CALL cb_remove(void *clientData) {
		if (locked) { return; }
		if (activeLayerCount > 1) {
			simMap->callbackRemove(selectedLayer);
			activeLayerCount -= 1;
			if (selectedLayer >= activeLayerCount) {
				unselectAll();
				selectedLayer = activeLayerCount - 1;
				lay[selectedLayer] = true;
			}
			refreshLayers();
		}
	}
	void TW_CALL SetNameStr(const void *value, void *clientData)
	{
		const std::string *srcPtr = static_cast<const std::string *>(value);
		simMap->getLayer(selectedLayer)->name = *srcPtr;
		std::string str = " LayerList/Layer" + std::to_string(selectedLayer + 1) + " label='" + srcPtr->c_str() + "' ";
		TwDefine(str.c_str());
	}
	void TW_CALL GetNameStr(void *value, void *clientData)
	{
		std::string *destPtr = static_cast<std::string *>(value);
		TwCopyStdStringToLibrary(*destPtr, simMap->getLayer(selectedLayer)->name);
	}


	//updates the list of
	void refreshLayers() {
		TwAddVarCB(twBar, "tmp", TW_TYPE_BOOLCPP, &setLayerCallback, &getLayerCallback, &indexlist[0], "group='Layer List'");
		std::string str = "Layer";
		for (int i = 1; i <= viscount; ++i) {
			str = "Layer" + std::to_string(i);
			const char *cstr = str.c_str();
			TwRemoveVar(twBar, cstr);
		}
		for (int i = activeLayerCount - 1; i >= 0; --i) {
			str = "Layer" + std::to_string(i + 1);
			std::string strparam = "group = 'Layer List' label = ' " + simMap->getLayer(i)->name + "'";
			const char *cstr = str.c_str();
			TwAddVarCB(twBar, cstr, TW_TYPE_BOOLCPP, &setLayerCallback, &getLayerCallback, &indexlist[i], strparam.c_str());
		}
		viscount = activeLayerCount;
		TwRemoveVar(twBar, "tmp");
		refreshLayerMaterial();
	}

	void genLayerPanel(SimMap* argsimMap) {
		simMap = argsimMap;
		twBar = TwNewBar("LayerList");
		TwDefine(" LayerList size='300 360' color='196 116 205 204' position='70 20' ");
		TwAddVarRO(twBar, "Selected Layer", TW_TYPE_UINT32, &selectedLayer,
			" min=0 max=2 step=1 keyIncr=l keyDecr=L help='Currently selected layer for editing operations' ");
		TwAddVarRW(twBar, "Show All Layers", TW_TYPE_BOOLCPP, &showall,
			" help='When ticked off only layers up to the selected one are shown.' ");
		TwAddSeparator(twBar, NULL, NULL);
		lay[0] = true;
		TwAddVarCB(twBar, "Layer1", TW_TYPE_BOOLCPP, &setLayerCallback, &getLayerCallback, &indexlist[0], "group='Layer List'");
		TwAddSeparator(twBar, NULL, NULL);
		TwAddVarCB(twBar, "stat0", TW_TYPE_STDSTRING, &SetNameStr, &GetNameStr, NULL, " label='Layer Name:' group='Layer Properties' ");
		TwAddButton(twBar, "stat1", NULL, NULL, " label='Material Name: ???' group='Layer Properties' ");
		TwAddVarCB(twBar, "stat2", TW_TYPE_INT32, &SetMaterial, &GetMaterial, NULL, " label='Layer Material:' group='Layer Properties' help='Refers to the material index within the material window.' ");
		TwAddButton(twBar, "moveup", cb_move, &RETVALSL[0], " label='Move layer up' group='Layer Actions' ");
		TwAddButton(twBar, "movedown", cb_move, &RETVALSL[1], " label='Move Layer down' group='Layer Actions' ");
		TwAddButton(twBar, "addup", cb_add, &RETVALSL[0], " label='Load layer from file above' group='Layer Actions' help='Shows a dialog for loading a layer from /data/ above the selected layer.'");
		TwAddButton(twBar, "adddown", cb_add, &RETVALSL[1], " label='Load layer from file bellow' group='Layer Actions' help='Shows a dialog for loading a layer from /data/ bellow the selected layer.'");
		TwAddButton(twBar, "addbup", cb_addb, &RETVALSL[0], " label='Add a blank layer above' group='Layer Actions' help='Adds a blank layer above the selected layer.' ");
		TwAddButton(twBar, "addbdown", cb_addb, &RETVALSL[1], " label='Add a blank layer bellow' group='Layer Actions' help='Adds a blank layer bellow the selected layer.' ");
		TwAddButton(twBar, "remove", cb_remove, NULL, " label='Remove layer' group='Layer Actions' help='Removes the selected layer.'");
		TwAddButton(twBar, "save", cb_save, NULL, " label='Save layer to file' group='Layer Actions' help='Saves the current layer to file under <layername>.bmp' ");
		refreshLayers();
	}

	int getSelectedLayer() {
		return selectedLayer;
	}

	bool getShowAll() {
		return showall;
	}

}