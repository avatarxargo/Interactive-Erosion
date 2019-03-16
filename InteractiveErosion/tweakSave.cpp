#include "tweakSave.h"

TwBar * tw_save;
std::string word;

void TW_CALL cb_accept(void *clientData) {
	TwDeleteBar(tw_save);
}

void TW_CALL cb_cancel(void *clientData) {
	TwDeleteBar(tw_save);
}

void saveNameDialog(SimMap* simMap) {
	bar = TwNewBar("saver");
	std::string title = " saver label='Save Layer";
	title += "' size='300 400' color='205 96 116 124' ";
	TwDefine(title.c_str());
	TwAddVarRW(bar, "width", TW_TYPE_INT32, &width, "label='Width'");
	TwAddVarRW(bar, "height", TW_TYPE_INT32, &height, "label='Height'");
	TwAddSeparator(bar, NULL, NULL);
	TwAddButton(bar, "Generate", tw_callback, NULL, NULL);
}