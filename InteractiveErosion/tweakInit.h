#pragma once
#include <AntTweakBar.h>
#include <vector>
#include <fstream>
#include <windows.h>
#include <string>
#include "SimMap.cuh"

namespace tweakInit {
	void genInitWizard(HANDLE handleSim, HANDLE handleGL, SimMap * target, void(*displayFunc)(unsigned int width, unsigned int height, unsigned int depth, float scale));
}