#pragma once
#include <AntTweakBar.h>
#include <vector>
#include <fstream>
#include <windows.h>
#include <string>

namespace tweakSimpleWizard {

	void genLayerWizard(char * path, void (cb_cancel)(), void (cb_accept)(char*), bool up);

	void read_directory(const std::string& name, std::vector<std::string>& v);

}