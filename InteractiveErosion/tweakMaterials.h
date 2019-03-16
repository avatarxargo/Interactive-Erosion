#pragma once
#include "SimMaterial.cuh"
#include "SimMap.cuh"
#include <string>
#include <AntTweakBar.h>
#include "tweakSimpleWizard.h"

namespace tweakMaterials {
	void genMaterialPanel(SimMap* simMap);

	/**the currently selected index in the list of SimLayers*/
	int getSelectedMaterial();
}