#pragma once
#include "SimMap.cuh"
#include <string>
#include <AntTweakBar.h>
#include "tweakWizard.h"

namespace tweakLayers {
	void genLayerPanel(SimMap* simMap);

	/**the currently selected index in the list of SimLayers*/
	int getSelectedLayer();

	/**Whether the show all checkbox is active*/
	bool getShowAll();

}