/*
 * Copyright (C) Jari Blom
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/ibvs_sim/ibvs_sim.h"
 * @author Jari Blom
 * Simulate Image Based Visual Servoing
 */

#ifndef IBVS_SIM_H
#define IBVS_SIM_H

#include "math/pprz_geodetic_double.h"
#include "modules/computer_vision/opticflow_module.h"

extern void ibvs_sim_init();
extern void ibvs_sim_periodic();

// Define Camera coordinate system
struct Coor_camera{
	int x; // With (0,0) at the center of the camera frame
	int y;
	int xv; // Coordinate in the virtual camera frame
	int yv; //
};

struct Tracked_object{
	struct Coor_camera *corner_loc;
	float L_matrix[2][3];
	bool ibvs_go;
	bool ibvs_go2;
	bool set_guidance;
	struct Coor_camera object_cg;
	// Previous values
  float roiw_prev;
  float roih_prev;
  float droiwdtw;
  float droihdth;
  float prev_vision_time;
  float vision_time;
  uint32_t ndt_vz;
  float v_zarr[];
};

/*
struct ResultToWrite{
  struct EnuCoor_f *enu;
  struct EnuCoor_f *venu;
  float *timestamp;
  // Timestamps at which something special happens
  float t_ibvs_go;
  float t_1;
  float t_2;
  float t_3;
  float t_landed;
  float *t_new_window;
  uint32_t n_reinits;
  // Variables to deduce the goal coordinate from
  struct EnuCoor_f *enu_start;
  struct Coor_camera goal_pixel;
  uint32_t img_width;
  uint32_t img_height;
  // Still need fov
  // Object size
  struct Coor_camera object_size;
};
*/

// Needed for settings
extern struct Tracked_object object_to_track;

#endif

