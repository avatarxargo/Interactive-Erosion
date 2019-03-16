#include "tweakSimpleWizard.h"

namespace tweakSimpleWizard {

	TwBar * tw_wizard;
	void(*tw_callbackCancel)();
	void(*tw_callbackAccept)(char*);
	std::vector<std::string> tw_files;

	void TW_CALL cb_loadLayer(void *clientData) {
		tw_callbackAccept((char*)clientData);
		TwDeleteBar(tw_wizard);
	}

	void TW_CALL cb_cancelInternal(void *clientData) {
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

	void genLayerWizard(char * path, void (cb_cancel)(), void (cb_accept)(char*), bool up) {
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
		listFiles(path);
		TwAddSeparator(tw_wizard, NULL, NULL);
		TwAddButton(tw_wizard, "Cancel", cb_cancelInternal, NULL, NULL);
	}
}