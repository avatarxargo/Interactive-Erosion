#pragma once

/**Gausian falloff function for all sorts of editing brushes. Stregth of selection on (x,y) coordinates from center (cx,cy).*/
__device__
float d_gaussFalloff(float x, float y, float cx, float cy, float mu, float sigma);