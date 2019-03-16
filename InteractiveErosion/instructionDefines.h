//Just a collective file for defines used throughout all the SimMap manipulating threads.

#define DW dd_intParams[0] //Width of the terrain
#define DH dd_intParams[1] //Height of the terrain
#define DIDX dd_intParams[2] //Index of the selected layer
#define DLEN dd_intParams[3] //Length of the SimLayer list
#define DPOLY_SEL_LEN dd_intParams[4] //Length of the polygon
#define DPOLY_SPR_LEN dd_intParams[5] //Length of the sprinklers

#define DSTR dd_floatParams[0] //Tool strength
#define DRAD dd_floatParams[1] //Tool radius
#define DX dd_floatParams[2] //Cursor x
#define DY dd_floatParams[3] //Cursor y
#define DZ dd_floatParams[4] //Cursor z
#define DSPRINKLER_STR dd_floatParams[5] //Sprinkler strength
#define DSPRINKLER_RADIUS dd_floatParams[6] //Sprinkler radius
#define DEVAPORATION dd_floatParams[7] //Evaporation rate

#define DUST dd_terrainData[DLEN - 1] //upper most layer
#define WATER dd_terrainData[DLEN] //current water level
#define MASK dd_terrainData[DLEN+1] //selection mask
#define SUMS dd_terrainData[DLEN+2] //sum of heights in each cell
#define WATER_LAST dd_terrainData[DLEN+3] //average of water levels
#define REGOLITH dd_terrainData[DLEN+4] //regolith level
#define SEDIMENT dd_terrainData[DLEN+5] //sediment level
#define MISCOBJ dd_terrainData[DLEN+6] //unused
#define WATER_VERT dd_terrainData[DLEN+7] //verttical pipe speeds
#define WATER_HOR dd_terrainData[DLEN+8] //horrizontal pipe speeds
#define WATER_CELL_VERT dd_terrainData[DLEN+9] //per-cell vert. spd
#define WATER_CELL_HOR dd_terrainData[DLEN+10] //per-cell hor. spd
#define SLOPE_SIN dd_terrainData[DLEN+11] //sin of the max slope
#define POLY_SEL dd_terrainData[DLEN+12] //list of polygon vertexes
#define POLY_SPR dd_terrainData[DLEN+13] //list of sprinkler vertexes

//material of the selected layer
#define CUR_MATERIAL d_materialData[d_materialIndex[DIDX]]
#define MATERIAL dd_materialIndex //layer -> material index
#define MDATA dd_materialData //list of materials
#define MTHERMAL [0] //thermal erosion rate
#define MANGLE [1] //talos angle
#define MHYDRO [2] //hydraulic erosion factor
#define MSEDIMENT [2] //rate of kinetic sattling
#define MKINETIC [3] //rate of kinetic erosion

#define CELL_WIDTH 1  //Cell width for water calculations