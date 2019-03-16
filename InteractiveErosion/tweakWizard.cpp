#include "tweakWizard.h"

namespace tweakWizard {

	TwBar * tw_wizard;
	void(*tw_callbackCancel)();
	void(*tw_callbackAccept)(char*, int, float);
	std::vector<std::string> tw_files;
	int tw_rgb = 0;
	float tw_scale = 1;
	bool tw_rgbool[4] = { true, false, false, false };
	const int tw_rgbval[4] = { 0,1,2,3 };

	void TW_CALL cb_loadLayer(void *clientData) {
		tw_callbackAccept((char*)clientData, tw_rgb, tw_scale);
		TwDeleteBar(tw_wizard);
	}

	void TW_CALL cb_cancelInternal(void *clientData) {
		//callback to original function.
		tw_callbackCancel();
		TwDeleteBar(tw_wizard);
	}

	//lists all loadable image files in provided directory with callbacks.
	void listFiles(char * path) {
		read_directory(path, tw_files);
		for (int i = 2; i < tw_files.size(); ++i) {
			TwAddButton(tw_wizard, tw_files[i].c_str(), cb_loadLayer, (void*)tw_files[i].c_str(), NULL);
		}
	}

	//
	void read_directory(const std::string& name, std::vector<std::string>& v)
	{
		v.clear();
		std::string pattern(name);
		pattern.append("\\*");
		WIN32_FIND_DATA data;
		HANDLE hFind;
		if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
			do {
				v.push_back(data.cFileName);
			} while (FindNextFile(hFind, &data) != 0);
			FindClose(hFind);
		}
	}

	void TW_CALL tw_setRgbCallback(const void *value, void *clientData)
	{
		if (*(const int *)value == false) { return; }
		tw_rgb = *(int *)clientData;
		tw_rgbool[0] = false;
		tw_rgbool[1] = false;
		tw_rgbool[2] = false;
		tw_rgbool[3] = false;
		tw_rgbool[*(int *)clientData] = true;
	}

	void TW_CALL tw_getRgbCallback(void *value, void *clientData)
	{
		*(bool *)value = tw_rgbool[*(int *)clientData];
	}

	void genLayerWizard(char * path, void (cb_cancel)(), void (cb_accept)(char*, int, float), bool up) {
		tw_callbackCancel = cb_cancel;
		tw_callbackAccept = cb_accept;

		tw_wizard = TwNewBar("newwiz");
		std::string title = " newwiz label='Add New Layer ";
		if (up) {
			title += "Above";
		}
		else {

			title += "Bellow";
		}
		title += "' size='300 400' color='205 96 116 124' ";
		TwDefine(title.c_str());
		TwAddSeparator(tw_wizard, NULL, NULL);
		TwAddVarCB(tw_wizard, "rgball", TW_TYPE_BOOLCPP, &tw_setRgbCallback, &tw_getRgbCallback, (void*)&tw_rgbval[0], "label='Load RGB Channels'");
		TwAddVarCB(tw_wizard, "rgbr", TW_TYPE_BOOLCPP, &tw_setRgbCallback, &tw_getRgbCallback, (void*)&tw_rgbval[1], "label='Load R Channel'");
		TwAddVarCB(tw_wizard, "rgbg", TW_TYPE_BOOLCPP, &tw_setRgbCallback, &tw_getRgbCallback, (void*)&tw_rgbval[2], "label='Load G Channel'");
		TwAddVarCB(tw_wizard, "rgbb", TW_TYPE_BOOLCPP, &tw_setRgbCallback, &tw_getRgbCallback, (void*)&tw_rgbval[3], "label='Load B Channel'");
		TwAddVarRW(tw_wizard, "scl", TW_TYPE_FLOAT, &tw_scale, "label='Vertical Scale'");
		TwAddSeparator(tw_wizard, NULL, NULL);
		TwAddButton(tw_wizard, "Cancel", cb_cancelInternal, NULL, NULL);
		TwAddSeparator(tw_wizard, NULL, NULL);
		listFiles(path);
	}
}