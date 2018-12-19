/*
 * Copyright (C) 2014 Hann Woei Ho
 *               2015 Freek van Tienen <freek.v.tienen@gmail.com>
 *               2016 Kimberly McGuire <k.n.mcguire@tudelft.nl
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

/**
 * @file modules/computer_vision/opticflow/opticflow_calculator.c
 * @brief Estimate velocity from optic flow.
 *
 */

#include "std.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Own Header
#include "opticflow_calculator.h"

// Computer Vision
#include "lib/vision/image.h"
#include "lib/vision/lucas_kanade.h"
#include "lib/vision/fast_rosten.h"
#include "lib/vision/act_fast.h"
#include "lib/vision/edge_flow.h"
#include "size_divergence.h"
#include "linear_flow_fit.h"
#include "modules/sonar/agl_dist.h"
// Include to check results
#include "state.h"
// for measuring time
#include "mcu_periph/sys_time.h"

// whether to show the flow and corners:
#define OPTICFLOW_SHOW_FLOW 1
#define OPTICFLOW_SHOW_CORNERS 1

#define EXHAUSTIVE_FAST 0
#define ACT_FAST 1
// TODO: these are now adapted, but perhaps later could be a setting:
uint16_t n_time_steps = 10;
uint16_t n_agents = 4;

// What methods are run to determine divergence, lateral flow, etc.
// SIZE_DIV looks at line sizes and only calculates divergence
#define SIZE_DIV 1
// LINEAR_FIT makes a linear optical flow field fit and extracts a lot of information:
// relative velocities in x, y, z (divergence / time to contact), the slope of the surface, and the surface roughness.
#define LINEAR_FIT 1

#ifndef OPTICFLOW_CORNER_METHOD
// This can be estimated by total possible image height / total Field of view
#define OPTICFLOW_CORNER_METHOD ACT_FAST
#endif
PRINT_CONFIG_VAR(OPTICFLOW_CORNER_METHOD)

// Camera parameters (defaults are from an ARDrone 2)
#ifndef OPTICFLOW_FOV_W
#define OPTICFLOW_FOV_W 0.89360857702
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FOV_W)

#ifndef OPTICFLOW_FOV_H
#define OPTICFLOW_FOV_H 0.67020643276
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FOV_H)

#ifndef OPTICFLOW_FX
// This can be estimated by total possible image width / total Field of view
#define OPTICFLOW_FX 343.1211
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FX)

#ifndef OPTICFLOW_FY
// This can be estimated by total possible image height / total Field of view
#define OPTICFLOW_FY 348.5053
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FY)

/* Set the default values */
#ifndef OPTICFLOW_MAX_TRACK_CORNERS
#define OPTICFLOW_MAX_TRACK_CORNERS 25
#endif
PRINT_CONFIG_VAR(OPTICFLOW_MAX_TRACK_CORNERS)

#ifndef OPTICFLOW_WINDOW_SIZE
#define OPTICFLOW_WINDOW_SIZE 10
#endif
PRINT_CONFIG_VAR(OPTICFLOW_WINDOW_SIZE)

#ifndef OPTICFLOW_SEARCH_DISTANCE
#define OPTICFLOW_SEARCH_DISTANCE 20
#endif
PRINT_CONFIG_VAR(OPTICFLOW_SEARCH_DISTANCE)

#ifndef OPTICFLOW_SUBPIXEL_FACTOR
#define OPTICFLOW_SUBPIXEL_FACTOR 10
#endif
PRINT_CONFIG_VAR(OPTICFLOW_SUBPIXEL_FACTOR)

#ifndef OPTICFLOW_RESOLUTION_FACTOR
#define OPTICFLOW_RESOLUTION_FACTOR 100
#endif
PRINT_CONFIG_VAR(OPTICFLOW_RESOLUTION_FACTOR)

#ifndef OPTICFLOW_MAX_ITERATIONS
#define OPTICFLOW_MAX_ITERATIONS 10
#endif
PRINT_CONFIG_VAR(OPTICFLOW_MAX_ITERATIONS)

#ifndef OPTICFLOW_THRESHOLD_VEC
#define OPTICFLOW_THRESHOLD_VEC 2
#endif
PRINT_CONFIG_VAR(OPTICFLOW_THRESHOLD_VEC)

#ifndef OPTICFLOW_PYRAMID_LEVEL
#define OPTICFLOW_PYRAMID_LEVEL 0
#endif
PRINT_CONFIG_VAR(OPTICFLOW_PYRAMID_LEVEL)

#ifndef OPTICFLOW_FAST9_ADAPTIVE
#define OPTICFLOW_FAST9_ADAPTIVE TRUE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FAST9_ADAPTIVE)

#ifndef OPTICFLOW_FAST9_THRESHOLD
#define OPTICFLOW_FAST9_THRESHOLD 20
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FAST9_THRESHOLD)

#ifndef OPTICFLOW_FAST9_MIN_DISTANCE
#define OPTICFLOW_FAST9_MIN_DISTANCE 10
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FAST9_MIN_DISTANCE)

#ifndef OPTICFLOW_FAST9_PADDING
#define OPTICFLOW_FAST9_PADDING 20
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FAST9_PADDING)

// thresholds FAST9 that are currently not set from the GCS:
#define FAST9_LOW_THRESHOLD 5
#define FAST9_HIGH_THRESHOLD 60

#ifndef OPTICFLOW_METHOD
#define OPTICFLOW_METHOD 0
#endif
PRINT_CONFIG_VAR(OPTICFLOW_METHOD)

#if OPTICFLOW_METHOD > 1
#error WARNING: Both Lukas Kanade and EdgeFlow are NOT selected
#endif

#ifndef OPTICFLOW_DEROTATION
#define OPTICFLOW_DEROTATION TRUE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_DEROTATION)

#ifndef OPTICFLOW_DEROTATION_CORRECTION_FACTOR_X
#define OPTICFLOW_DEROTATION_CORRECTION_FACTOR_X 1.0
#endif
PRINT_CONFIG_VAR(OPTICFLOW_DEROTATION_CORRECTION_FACTOR_X)

#ifndef OPTICFLOW_DEROTATION_CORRECTION_FACTOR_Y
#define OPTICFLOW_DEROTATION_CORRECTION_FACTOR_Y 1.0
#endif
PRINT_CONFIG_VAR(OPTICFLOW_DEROTATION_CORRECTION_FACTOR_Y)

#ifndef OPTICFLOW_MEDIAN_FILTER
#define OPTICFLOW_MEDIAN_FILTER FALSE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_MEDIAN_FILTER)

#ifndef OPTICFLOW_FEATURE_MANAGEMENT
#define OPTICFLOW_FEATURE_MANAGEMENT 1
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FEATURE_MANAGEMENT)

#ifndef OPTICFLOW_FAST9_REGION_DETECT
#define OPTICFLOW_FAST9_REGION_DETECT 1
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FAST9_REGION_DETECT)

#ifndef OPTICFLOW_FAST9_NUM_REGIONS
#define OPTICFLOW_FAST9_NUM_REGIONS 9
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FAST9_NUM_REGIONS)

#ifndef OPTICFLOW_ACTFAST_LONG_STEP
#define OPTICFLOW_ACTFAST_LONG_STEP 10
#endif
PRINT_CONFIG_VAR(OPTICFLOW_ACTFAST_LONG_STEP)

#ifndef OPTICFLOW_ACTFAST_SHORT_STEP
#define OPTICFLOW_ACTFAST_SHORT_STEP 2
#endif
PRINT_CONFIG_VAR(OPTICFLOW_ACTFAST_SHORT_STEP)

#ifndef OPTICFLOW_ACTFAST_LONG_STEP_OBJECT
#define OPTICFLOW_ACTFAST_LONG_STEP_OBJECT 3
#endif
PRINT_CONFIG_VAR(OPTICFLOW_ACTFAST_LONG_STEP_OBJECT)

#ifndef OPTICFLOW_ACTFAST_SHORT_STEP_OBJECT
#define OPTICFLOW_ACTFAST_SHORT_STEP_OBJECT 2
#endif
PRINT_CONFIG_VAR(OPTICFLOW_ACTFAST_SHORT_STEP_OBJECT)

#ifndef OPTICFLOW_ACTFAST_GRADIENT_METHOD
#define OPTICFLOW_ACTFAST_GRADIENT_METHOD 1
#endif
PRINT_CONFIG_VAR(OPTICFLOW_ACTFAST_GRADIENT_METHOD)

#ifndef OPTICFLOW_ACTFAST_MIN_GRADIENT
#define OPTICFLOW_ACTFAST_MIN_GRADIENT 10
#endif
PRINT_CONFIG_VAR(OPTICFLOW_ACTFAST_MIN_GRADIENT)

#ifndef OPTICFLOW_OBJECT_TRACKING
#define OPTICFLOW_OBJECT_TRACKING FALSE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_OBJECT_TRACKING)

#ifndef NR_OF_CORNERS_TO_TRACK
#define NR_OF_CORNERS_TO_TRACK 4
#endif
PRINT_CONFIG_VAR(NR_OF_CORNERS_TO_TRACK)

#ifndef OPTICFLOW_IBVS_INIT
#define OPTICFLOW_IBVS_INIT FALSE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_IBVS_INIT)

#ifndef OPTICFLOW_SHAPE_CORRECT
#define OPTICFLOW_SHAPE_CORRECT FALSE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_SHAPE_CORRECT)


// Defaults for ARdrone
#ifndef OPTICFLOW_BODY_TO_CAM_PHI
#define OPTICFLOW_BODY_TO_CAM_PHI 0
#endif
#ifndef OPTICFLOW_BODY_TO_CAM_THETA
#define OPTICFLOW_BODY_TO_CAM_THETA 0
#endif
#ifndef OPTICFLOW_BODY_TO_CAM_PSI
#define OPTICFLOW_BODY_TO_CAM_PSI -M_PI_2
#endif

//Include median filter
#include "filters/median_filter.h"
struct MedianFilter3Float vel_filt;
struct FloatRMat body_to_cam;
static uint16_t run_count = 0;


// Define max and min functions
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* Functions only used here */
static uint32_t timeval_diff(struct timeval *starttime, struct timeval *finishtime);
static int cmp_flow(const void *a, const void *b);
static int cmp_array(const void *a, const void *b);
static void manage_flow_features(struct opticflow_t *opticflow, struct opticflow_result_t *result, uint16_t *roi);
static void manage_flow_features_object(struct opticflow_t *opticflow, struct opticflow_result_t *result);
static void init_object_tracking(struct opticflow_t *opticflow);
static void check_for_features(struct opticflow_t *opticflow, struct opticflow_result_t *result);

/**
 * Initialize the opticflow calculator
 * @param[out] *opticflow The new optical flow calculator
 */
void opticflow_calc_init(struct opticflow_t *opticflow)
{
  /* Set the default values */
  opticflow->method = OPTICFLOW_METHOD; //0 = LK_fast9, 1 = Edgeflow
  opticflow->window_size = OPTICFLOW_WINDOW_SIZE;
  opticflow->search_distance = OPTICFLOW_SEARCH_DISTANCE;
  opticflow->derotation = OPTICFLOW_DEROTATION; //0 = OFF, 1 = ON
  opticflow->derotation_correction_factor_x = OPTICFLOW_DEROTATION_CORRECTION_FACTOR_X;
  opticflow->derotation_correction_factor_y = OPTICFLOW_DEROTATION_CORRECTION_FACTOR_Y;

  opticflow->max_track_corners = OPTICFLOW_MAX_TRACK_CORNERS;
  opticflow->subpixel_factor = OPTICFLOW_SUBPIXEL_FACTOR;
  if (opticflow->subpixel_factor == 0) {
    opticflow->subpixel_factor = 10;
  }
  opticflow->resolution_factor = OPTICFLOW_RESOLUTION_FACTOR;
  opticflow->max_iterations = OPTICFLOW_MAX_ITERATIONS;
  opticflow->threshold_vec = OPTICFLOW_THRESHOLD_VEC;
  opticflow->pyramid_level = OPTICFLOW_PYRAMID_LEVEL;
  opticflow->median_filter = OPTICFLOW_MEDIAN_FILTER;
  opticflow->feature_management = OPTICFLOW_FEATURE_MANAGEMENT;
  opticflow->fast9_region_detect = OPTICFLOW_FAST9_REGION_DETECT;
  opticflow->fast9_num_regions = OPTICFLOW_FAST9_NUM_REGIONS;

  opticflow->fast9_adaptive = OPTICFLOW_FAST9_ADAPTIVE;
  opticflow->fast9_threshold = OPTICFLOW_FAST9_THRESHOLD;
  opticflow->fast9_min_distance = OPTICFLOW_FAST9_MIN_DISTANCE;
  opticflow->fast9_padding = OPTICFLOW_FAST9_PADDING;
  opticflow->fast9_rsize = 512;
  opticflow->fast9_ret_corners = calloc(opticflow->fast9_rsize, sizeof(struct point_tf));

  opticflow->corner_method = OPTICFLOW_CORNER_METHOD;
  opticflow->actfast_long_step = OPTICFLOW_ACTFAST_LONG_STEP;
  opticflow->actfast_short_step = OPTICFLOW_ACTFAST_SHORT_STEP;
  opticflow->actfast_min_gradient = OPTICFLOW_ACTFAST_MIN_GRADIENT;
  opticflow->actfast_gradient_method = OPTICFLOW_ACTFAST_GRADIENT_METHOD;

  struct FloatEulers euler = {OPTICFLOW_BODY_TO_CAM_PHI, OPTICFLOW_BODY_TO_CAM_THETA, OPTICFLOW_BODY_TO_CAM_PSI};
  float_rmat_of_eulers(&body_to_cam, &euler);

  opticflow->object_tracking = OPTICFLOW_OBJECT_TRACKING;
  opticflow->object_tracking_set = false;
  opticflow->ibvs_init = OPTICFLOW_IBVS_INIT;

  opticflow->previous_fast9_ret_corners = calloc(opticflow->fast9_rsize, sizeof(struct point_tf));
  opticflow->n_new_corners = 0;

  opticflow->nr_of_new_corners_to_detect = 25; // Might have to be increased back to 25
  opticflow->new_fast9_ret_corners = calloc(opticflow->nr_of_new_corners_to_detect, sizeof(struct point_tf));
  opticflow->object_tracking_reset = false;
  opticflow->in_flight = true;
  opticflow->t_new_window = calloc(30, sizeof(float));
  opticflow->n_reinits = 0;
  opticflow->roih_init = 80.0;
  opticflow->roiw_init = 60.0;
  opticflow->roi_center_init.x = 130;
  opticflow->roi_center_init.y = 120;

  // Initialize logger
  uint32_t file_counter = 0;
  char filename[512];
  char* ibvs_file_logger_path = "./data/Jari";
  opticflow->ibvs_file_logger = NULL;
  // Check for available files
  sprintf(filename, "%s/run%d.csv", ibvs_file_logger_path, (int32_t)file_counter);
  while ((opticflow->ibvs_file_logger = fopen(filename, "r"))) {
    fclose(opticflow->ibvs_file_logger);

    file_counter++;
    sprintf(filename, "%s/run%d.csv", ibvs_file_logger_path, (int32_t)file_counter);
  }
  printf("%s\n",filename);
  opticflow->ibvs_file_logger = fopen(filename,"w");
  if(opticflow->ibvs_file_logger != NULL){
    fprintf(opticflow->ibvs_file_logger,"t\tx\ty\tz\tvx\tvy\tvz\txstart\tystart\tzstart\troihinit\troiwinit\troixinit"
      "\troiyinit\timgw\timgh\tfov\tgoaldiv\tt0\tt1\tt2\tt3\ttlanded\tnreinits\tphistart\tthetastart\tpsistart\n");
  }
}
/**
 * Run the optical flow with fast9 and lukaskanade on a new image frame
 * @param[in] *opticflow The opticalflow structure that keeps track of previous images
 * @param[in] *state The state of the drone
 * @param[in] *img The image frame to calculate the optical flow from
 * @param[out] *result The optical flow result
 * @return Was optical flow successful
 */
bool calc_fast9_lukas_kanade(struct opticflow_t *opticflow, struct image_t *img,
                             struct opticflow_result_t *result)
{
  if (opticflow->just_switched_method) {
    // Create the image buffers
    image_create(&opticflow->img_gray, img->w, img->h, IMAGE_GRAYSCALE);
    image_create(&opticflow->prev_img_gray, img->w, img->h, IMAGE_GRAYSCALE);

    // Set the previous values
    opticflow->got_first_img = false;

    // Init median filters with zeros
    InitMedianFilterVect3Float(vel_filt, MEDIAN_DEFAULT_SIZE);

    result->corner_cnt = 0;
    result->tracked_cnt = 0;
  }

  // Convert image to grayscale
  image_to_grayscale(img, &opticflow->img_gray);
  opticflow->img_gray.ts = img->ts;
  opticflow->img_gray.pprz_ts = img->pprz_ts;
  opticflow->img_gray.eulers = img->eulers;

  // Copy to previous image if not set
  if (!opticflow->got_first_img) {
    image_copy(&opticflow->img_gray, &opticflow->prev_img_gray);
    opticflow->got_first_img = true;
    return false;
  }

  // variables for linear flow fit:
  float error_threshold;
  int n_iterations_RANSAC, n_samples_RANSAC, success_fit;
  struct linear_flow_fit_info fit_info;

  // Update FPS for information
  float dt = timeval_diff(&(opticflow->prev_img_gray.ts), &(img->ts)) / 1000.f;
  if (dt > 1e-5) {
    result->fps = 1 / dt;
  } else {
    return false;
  }

  // *************************************************************************************
  // Corner detection
  // *************************************************************************************

  // if feature_management is selected and tracked corners drop below a threshold, redetect
  if (opticflow->feature_management) {
    result->corner_cnt = result->tracked_cnt;
    // check if new region of interest defined
    if (opticflow->object_tracking && opticflow->object_tracking_set){
      printf("Reset \n");
      init_object_tracking(opticflow);
      result->corner_cnt = 0;
      manage_flow_features(opticflow, result, opticflow->roi);
      for (uint16_t i = 0; i < result->corner_cnt; i++){
        opticflow->fast9_ret_corners[i].x_full = opticflow->fast9_ret_corners[i].x * opticflow->subpixel_factor;
        opticflow->fast9_ret_corners[i].y_full = opticflow->fast9_ret_corners[i].y * opticflow->subpixel_factor;
      }
    }
    // Look for new features in the ROI
    else if(opticflow->object_tracking && result->corner_cnt < opticflow->max_track_corners-2){
      // define number of corners to look for
      opticflow->n_new_corners = opticflow->max_track_corners - result->corner_cnt;
      check_for_features(opticflow, result);
    }
	  // update feature management
    else if (result->corner_cnt < opticflow->max_track_corners / 2){
	    result->corner_cnt = 0;
		  manage_flow_features(opticflow, result, NULL);
		  for (uint16_t i = 0; i < result->corner_cnt; i++){
		    opticflow->fast9_ret_corners[i].x_full = opticflow->fast9_ret_corners[i].x * opticflow->subpixel_factor;
		    opticflow->fast9_ret_corners[i].y_full = opticflow->fast9_ret_corners[i].y * opticflow->subpixel_factor;
		  }
		  /*opticflow->object_tracking_set = false;*/
	  }
  } else {
    // needs to be set to 0 because result is now static
    result->corner_cnt = 0;

    if (opticflow->corner_method == EXHAUSTIVE_FAST) {
      // FAST corner detection
      // TODO: There is something wrong with fast9_detect destabilizing FPS. This problem is reduced with putting min_distance
      // to 0 (see defines), however a more permanent solution should be considered
      fast9_detect(&opticflow->prev_img_gray, opticflow->fast9_threshold, opticflow->fast9_min_distance,
                   opticflow->fast9_padding, opticflow->fast9_padding, &result->corner_cnt,
                   &opticflow->fast9_rsize,
                   &opticflow->fast9_ret_corners,
                   NULL);

    } else if (opticflow->corner_method == ACT_FAST) {
      // ACT-FAST corner detection:
      act_fast(&opticflow->prev_img_gray, opticflow->fast9_threshold, &result->corner_cnt,
               &opticflow->fast9_ret_corners, n_agents, n_time_steps,
               opticflow->actfast_long_step, opticflow->actfast_short_step, opticflow->actfast_min_gradient,
               opticflow->actfast_gradient_method);
    }
    // Adaptive threshold
    if (opticflow->fast9_adaptive) {

      // This works well for exhaustive FAST, but drives the threshold to the minimum for ACT-FAST:
      // Decrease and increase the threshold based on previous values
      if (result->corner_cnt < 40) { // TODO: Replace 40 with OPTICFLOW_MAX_TRACK_CORNERS / 2
        // make detections easier:
        if (opticflow->fast9_threshold > FAST9_LOW_THRESHOLD) {
          opticflow->fast9_threshold--;
        }

        if (opticflow->corner_method == ACT_FAST) {
          n_time_steps++;
          n_agents++;
        }

      } else if (result->corner_cnt > OPTICFLOW_MAX_TRACK_CORNERS * 2 && opticflow->fast9_threshold < FAST9_HIGH_THRESHOLD) {
        opticflow->fast9_threshold++;
        if (opticflow->corner_method == ACT_FAST && n_time_steps > 5 && n_agents > 10) {
          n_time_steps--;
          n_agents--;
        }
      }
    }
  }

#if OPTICFLOW_SHOW_CORNERS
  static uint8_t white[4] = {127, 255, 127, 255};
  image_show_points_color(img, opticflow->fast9_ret_corners, result->corner_cnt, white);
#endif

  // Check if we found some corners to track
  if (result->corner_cnt < 1) {
    // Clear the result otherwise the previous values will be returned for this frame too
    VECT3_ASSIGN(result->vel_cam, 0, 0, 0);
    VECT3_ASSIGN(result->vel_body, 0, 0, 0);
    result->div_size = 0; result->divergence = 0;
    result->noise_measurement = 5.0;

    image_switch(&opticflow->img_gray, &opticflow->prev_img_gray);
    return false;
  }

  // *************************************************************************************
  // Corner Tracking
  // *************************************************************************************
  // Execute a Lucas Kanade optical flow
  result->tracked_cnt = result->corner_cnt;
  struct flow_t *vectors = opticFlowLK(&opticflow->img_gray, &opticflow->prev_img_gray, opticflow->fast9_ret_corners,
                                       &result->tracked_cnt,
                                       opticflow->window_size / 2, opticflow->subpixel_factor, opticflow->max_iterations,
                                       opticflow->threshold_vec, opticflow->max_track_corners, opticflow->pyramid_level);


  // *************************************************************************************
  // Next Loop Preparation
  // *************************************************************************************
  if (opticflow->feature_management) {
    result->corner_cnt = result->tracked_cnt;
    opticflow->nr_of_corners_detected = result->corner_cnt;
    // Calculate the new positions 1 decimal accurate to save for the next calculation
    // However use the integer version to display
    uint16_t local_roi[4];
    for (uint16_t i = 0; i < result->tracked_cnt; i++) {
      opticflow->fast9_ret_corners[i].x_full = vectors[i].pos.x + vectors[i].flow_x;
      opticflow->fast9_ret_corners[i].y_full = vectors[i].pos.y + vectors[i].flow_y;
      // round to nearest for int position
      opticflow->fast9_ret_corners[i].x = (uint32_t)roundf((float)opticflow->fast9_ret_corners[i].x_full/opticflow->subpixel_factor);
      opticflow->fast9_ret_corners[i].y = (uint32_t)roundf((float)opticflow->fast9_ret_corners[i].y_full/opticflow->subpixel_factor);
      opticflow->fast9_ret_corners[i].count = vectors[i].pos.count;

      uint8_t window = 2;
      if (opticflow->fast9_ret_corners[i].x <= window || opticflow->fast9_ret_corners[i].y <= window ||
          opticflow->fast9_ret_corners[i].x + window > img->w || opticflow->fast9_ret_corners[i].y + window > img->h){
        continue;
      }
      local_roi[0] = opticflow->fast9_ret_corners[i].x - window;
      local_roi[1] = opticflow->fast9_ret_corners[i].y - window;
      local_roi[2] = opticflow->fast9_ret_corners[i].x + window;
      local_roi[3] = opticflow->fast9_ret_corners[i].y + window;

      uint16_t local_corners_size = 32;
      struct point_tf *local_corners = calloc(local_corners_size, sizeof(struct point_tf));
      uint16_t local_corners_found = 0;
      fast9_detect(&opticflow->img_gray, opticflow->fast9_threshold, 1,
                           0, 0, &local_corners_found,
                           &local_corners_size,
                           &local_corners,
                           local_roi);

      if (local_corners_found == 0)
      {
        // Reduce result->tracked_cnt
        result->tracked_cnt -= 1;
        // Remove entry from vectors
        // Move entry at tracked_cnt to this location
        vectors[i].pos.x = vectors[result->tracked_cnt].pos.x;
        vectors[i].pos.y = vectors[result->tracked_cnt].pos.y;
        vectors[i].flow_x = vectors[result->tracked_cnt].flow_x;
        vectors[i].flow_y = vectors[result->tracked_cnt].flow_y;
        // Remove entry from fast9_corners
        opticflow->fast9_ret_corners[i].x = opticflow->fast9_ret_corners[result->tracked_cnt].x;
        opticflow->fast9_ret_corners[i].y = opticflow->fast9_ret_corners[result->tracked_cnt].y;
        opticflow->fast9_ret_corners[i].x_full = opticflow->fast9_ret_corners[result->tracked_cnt].x_full;
        opticflow->fast9_ret_corners[i].y_full = opticflow->fast9_ret_corners[result->tracked_cnt].y_full;
        // Reduce i to make sure this corner is still checked
        i -= 1;

      } else {
        float min_dist = window * window;
        uint16_t min_ind = 0;
        for(uint16_t j = 0; j < local_corners_found; j++){
          float diff_x = (float)opticflow->fast9_ret_corners[i].x - (float)local_corners[j].x;
          float diff_y = (float)opticflow->fast9_ret_corners[i].y - (float)local_corners[j].y;
          float dist = sqrtf(diff_x*diff_x + diff_y*diff_y);
          if (dist < min_dist){
            min_dist = dist;
            min_ind = j;
          }
        }
        // update vectors with local_corners[min_ind].x - opticflow->fast9_ret_corners[i].x;
        vectors[i].flow_x += local_corners[min_ind].x*opticflow->subpixel_factor - opticflow->fast9_ret_corners[i].x_full;
        vectors[i].flow_y += local_corners[min_ind].y*opticflow->subpixel_factor - opticflow->fast9_ret_corners[i].y_full;

        // Update fast corners
        opticflow->fast9_ret_corners[i].x = local_corners[min_ind].x;
        opticflow->fast9_ret_corners[i].y = local_corners[min_ind].y;
        opticflow->fast9_ret_corners[i].x_full = opticflow->fast9_ret_corners[i].x * opticflow->subpixel_factor;
        opticflow->fast9_ret_corners[i].y_full = opticflow->fast9_ret_corners[i].y * opticflow->subpixel_factor;
      }

      free(local_corners);
    }

/*
#if OPTICFLOW_SHOW_CORNERS
    image_show_points(img, opticflow->fast9_ret_corners, result->corner_cnt);
#endif
*/
  }

#if OPTICFLOW_SHOW_FLOW
  image_show_flow(img, vectors, result->tracked_cnt, opticflow->subpixel_factor);
#endif

  // Global flow calculations
  static int n_samples = 100;
  // Estimate size divergence:
  if (SIZE_DIV) {
    result->div_size = get_size_divergence(vectors, result->tracked_cnt, n_samples);// * result->fps;

  } else {
    result->div_size = 0.0f;
  }

  if (LINEAR_FIT) {
    // Linear flow fit (normally derotation should be performed before):
    error_threshold = 10.0f;
    n_iterations_RANSAC = 20;
    n_samples_RANSAC = 5;
    success_fit = analyze_linear_flow_field(vectors, result->tracked_cnt, error_threshold, n_iterations_RANSAC,
                                            n_samples_RANSAC, img->w, img->h, &fit_info);

    if (!success_fit) {
      fit_info.divergence = 0.0f;
      fit_info.surface_roughness = 0.0f;
    }

    result->divergence = fit_info.divergence;
    result->surface_roughness = fit_info.surface_roughness;
  } else {
    result->divergence = 0.0f;
    result->surface_roughness = 0.0f;
  }

  // compute flow as mean of vectors
  int32_t flow_sum_x = 0, flow_sum_y = 0;
  for (uint16_t i = 0; i < result->tracked_cnt; i++){
    flow_sum_x += vectors[i].flow_x;
    flow_sum_y += vectors[i].flow_y;
  }
  result->flow_x = flow_sum_x / result->tracked_cnt;
  result->flow_y = flow_sum_y / result->tracked_cnt;

  // TODO scale flow to rad/s here

  // Flow Derotation
  float diff_flow_x = 0.f;
  float diff_flow_y = 0.f;

  if (opticflow->derotation && result->tracked_cnt > 5) {
    diff_flow_x = (opticflow->img_gray.eulers.phi - opticflow->prev_img_gray.eulers.phi) * OPTICFLOW_FX;
    diff_flow_y = (opticflow->img_gray.eulers.theta - opticflow->prev_img_gray.eulers.theta) * OPTICFLOW_FY;
    /*diff_flow_x = (cam_state->rates.p)  / result->fps * img->w /
                  OPTICFLOW_FOV_W;// * img->w / OPTICFLOW_FOV_W;
    diff_flow_y = (cam_state->rates.q) / result->fps * img->h /
                  OPTICFLOW_FOV_H;// * img->h / OPTICFLOW_FOV_H;*/
  }

  float rotation_threshold = M_PI / 180.0f;
  if (fabs(opticflow->img_gray.eulers.phi - opticflow->prev_img_gray.eulers.phi) > rotation_threshold) {
    result->flow_der_x = 0.0f;
    result->flow_der_y = 0.0f;
  } else {
    result->flow_der_x = result->flow_x - diff_flow_x * opticflow->subpixel_factor *
                         opticflow->derotation_correction_factor_x;
    result->flow_der_y = result->flow_y - diff_flow_y * opticflow->subpixel_factor *
                         opticflow->derotation_correction_factor_y;
  }

  // Velocity calculation
  // Right now this formula is under assumption that the flow only exist in the center axis of the camera.
  // TODO Calculate the velocity more sophisticated, taking into account the drone's angle and the slope of the ground plane.
  result->vel_cam.x = (float)result->flow_der_x * result->fps * agl_dist_value_filtered /
                      (opticflow->subpixel_factor * OPTICFLOW_FX);
  result->vel_cam.y = (float)result->flow_der_y * result->fps * agl_dist_value_filtered /
                      (opticflow->subpixel_factor * OPTICFLOW_FY);
  result->vel_cam.z = result->divergence * result->fps * agl_dist_value_filtered;

  //Apply a  median filter to the velocity if wanted
  if (opticflow->median_filter == true) {
    UpdateMedianFilterVect3Float(vel_filt, result->vel_cam);
  }

  // Update ROI
  if(opticflow->object_tracking){
    // Before we move ROI we should do two checks
    // 1. Did we initialize offset_roi_cg already?
    if (!opticflow->offset_defined){
      // Keep track of previous corners for future reference
      opticflow->previous_tracked_cnt = result->tracked_cnt;
      for (uint16_t i = 0; i < result->tracked_cnt; i++) {
        opticflow->previous_fast9_ret_corners[i].x_full = opticflow->fast9_ret_corners[i].x_full;
        opticflow->previous_fast9_ret_corners[i].y_full = opticflow->fast9_ret_corners[i].y_full;
        opticflow->previous_fast9_ret_corners[i].x = opticflow->fast9_ret_corners[i].x;
        opticflow->previous_fast9_ret_corners[i].y = opticflow->fast9_ret_corners[i].y;
      }
      // Find average position of corners
      struct point_tf av_corn_pos;
      av_corn_pos.x_full = 0, av_corn_pos.y_full = 0;
      for (uint16_t i = 0; i < result->tracked_cnt; i++) {
        av_corn_pos.x_full += opticflow->fast9_ret_corners[i].x_full;
        av_corn_pos.y_full += opticflow->fast9_ret_corners[i].y_full;
      }
      opticflow->cg_corners.x_full = av_corn_pos.x_full/result->tracked_cnt;
      opticflow->cg_corners.y_full = av_corn_pos.y_full/result->tracked_cnt;
      // Find correction: roi - cg
      opticflow->offset_roi_cg.x_full = (int32_t)opticflow->roi_center.x_full - (int32_t)opticflow->cg_corners.x_full;
      opticflow->offset_roi_cg.y_full = (int32_t)opticflow->roi_center.y_full - (int32_t)opticflow->cg_corners.y_full;
      opticflow->offset_defined = true;
      // Find average distance to the cg
      struct point_tf av_dist; av_dist.x_full = 0; av_dist.y_full = 0;
      for (uint16_t i = 0; i < result->tracked_cnt; i++) {
        av_dist.x_full += abs((int32_t)opticflow->fast9_ret_corners[i].x_full-(int32_t)opticflow->cg_corners.x_full);
        av_dist.y_full += abs((int32_t)opticflow->fast9_ret_corners[i].y_full-(int32_t)opticflow->cg_corners.y_full);
      }
      opticflow->av_dist.x_full = av_dist.x_full/result->tracked_cnt;
      opticflow->av_dist.y_full = av_dist.y_full/result->tracked_cnt;

      // Define initial w/d and h/d
      opticflow->width_dist_ratio = opticflow->roiw*(float)(opticflow->subpixel_factor)/opticflow->av_dist.x_full;
      opticflow->height_dist_ratio = opticflow->roih*(float)(opticflow->subpixel_factor)/opticflow->av_dist.y_full;
      opticflow->n_new_corners = 0;
    } else{
      // Check if we were able to keep at least 15 of the old corners
      // Three times in a row keeping less than 15 corners, means we can assume we're in the last phase of the landing

      // 2. Did we lose corners?
      if (opticflow->previous_tracked_cnt > result->tracked_cnt){
        // Nr of lost corners
        uint8_t cnt_diff = opticflow->previous_tracked_cnt - result->tracked_cnt;
        // Array of removed corner index
        uint16_t rem_ind[cnt_diff];
        for (uint16_t j = 0; j < cnt_diff; j++) {
          struct point_tf nr_to_look_for;
          // Look for the index in the new list of the entry most closely matching the last entry of the previous list
          nr_to_look_for.x = opticflow->previous_fast9_ret_corners[opticflow->previous_tracked_cnt-j-1].x_full;
          nr_to_look_for.y = opticflow->previous_fast9_ret_corners[opticflow->previous_tracked_cnt-j-1].y_full;
          float min_dist = 100;
          uint16_t min_ind = 0;
          for (uint16_t i = 0; i < result->tracked_cnt; i++) {
            float diff_x = (float)opticflow->fast9_ret_corners[i].x_full - (float)nr_to_look_for.x;
            float diff_y = (float)opticflow->fast9_ret_corners[i].y_full - (float)nr_to_look_for.y;
            float dist = sqrtf(diff_x*diff_x + diff_y*diff_y);
            if (dist < min_dist){
              min_dist = dist;
              min_ind = i;
            }
          }
          rem_ind[j] = min_ind;
        }
        // Find lost distance due to lost corner
        struct point_tf lost_corn_dist_sum;
        lost_corn_dist_sum.x_full = 0; lost_corn_dist_sum.y_full = 0;
        for (uint16_t j = 0; j < cnt_diff; j++) {
          lost_corn_dist_sum.x_full += abs((int32_t)opticflow->previous_fast9_ret_corners[rem_ind[j]].x_full-(int32_t)opticflow->cg_corners.x_full);
          lost_corn_dist_sum.y_full += abs((int32_t)opticflow->previous_fast9_ret_corners[rem_ind[j]].y_full-(int32_t)opticflow->cg_corners.y_full);
        }
        // Find lost coordinates due to lost corner
        struct point_tf lost_corn_sum;
        lost_corn_sum.x_full = 0; lost_corn_sum.y_full = 0;
        for (uint16_t j = 0; j < cnt_diff; j++) {
          lost_corn_sum.x_full += opticflow->previous_fast9_ret_corners[rem_ind[j]].x_full;
          lost_corn_sum.y_full += opticflow->previous_fast9_ret_corners[rem_ind[j]].y_full;
        }
        // Calculate w/d correction
        float w_corr_lost_x = (opticflow->av_dist.x_full*(float)result->tracked_cnt)/
            (opticflow->av_dist.x_full*(float)opticflow->previous_tracked_cnt-(float)lost_corn_dist_sum.x_full);
        float h_corr_lost_y = (opticflow->av_dist.y_full*(float)result->tracked_cnt)/
            (opticflow->av_dist.y_full*(float)opticflow->previous_tracked_cnt-(float)lost_corn_dist_sum.y_full);
        // Update w/d
        opticflow->width_dist_ratio *= w_corr_lost_x;
        opticflow->height_dist_ratio *= h_corr_lost_y;
        // Calculate roi_cg correction
        struct point_tf_signed off_cor_lost;
        float old_new_corr_ratio = (float)opticflow->previous_tracked_cnt/(float)result->tracked_cnt -1.0;
        off_cor_lost.x_full = (int32_t)roundf((old_new_corr_ratio*(float)opticflow->cg_corners.x_full)-((float)lost_corn_sum.x_full/(float)result->tracked_cnt));
        off_cor_lost.y_full = (int32_t)roundf((old_new_corr_ratio*(float)opticflow->cg_corners.y_full)-((float)lost_corn_sum.y_full/(float)result->tracked_cnt));
        // Update roi cg offset
        opticflow->offset_roi_cg.x_full -= off_cor_lost.x_full;
        opticflow->offset_roi_cg.y_full -= off_cor_lost.y_full;
      }
      // When new corners are detected, we update the roi parameters
      if (opticflow->n_new_corners>0){
        // New corners are at: tracked_cnt and later
        struct point_tf found_sum; found_sum.x_full = 0; found_sum.y_full = 0;
        for(uint16_t i=0;i<opticflow->n_new_corners;i++){
          found_sum.x_full += opticflow->fast9_ret_corners[i+result->tracked_cnt].x_full;
          found_sum.y_full += opticflow->fast9_ret_corners[i+result->tracked_cnt].y_full;
        }
        // New number of corners
        uint16_t n_new = opticflow->n_new_corners + opticflow->previous_tracked_cnt;
        // Correct offset
        struct point_tf_signed off_cor_found;
        float mult_fact_old_new = (float)opticflow->previous_tracked_cnt/(float)n_new-1.0;
        off_cor_found.x_full = (int32_t)roundf((float)found_sum.x_full/(float)n_new+(float)opticflow->cg_corners.x_full*mult_fact_old_new);
        off_cor_found.y_full = (int32_t)roundf((float)found_sum.y_full/(float)n_new+(float)opticflow->cg_corners.y_full*mult_fact_old_new);
        opticflow->offset_roi_cg.x_full -= off_cor_found.x_full;
        opticflow->offset_roi_cg.y_full -= off_cor_found.y_full;
        // Correct w/d
        struct point_tfloat width_corr_found;
        struct point_tf found_dist_sum; found_dist_sum.x_full = 0; found_dist_sum.y_full = 0;
        for(uint16_t i=0;i<opticflow->n_new_corners;i++){
          found_dist_sum.x_full += abs((int)opticflow->fast9_ret_corners[i+result->tracked_cnt].x_full-(int)opticflow->cg_corners.x_full);
          found_dist_sum.y_full += abs((int)opticflow->fast9_ret_corners[i+result->tracked_cnt].y_full-(int)opticflow->cg_corners.y_full);
        }
        width_corr_found.x_full = opticflow->av_dist.x_full*((float)n_new)/
            (opticflow->av_dist.x_full*(float)opticflow->previous_tracked_cnt + (float)found_dist_sum.x_full);
        width_corr_found.y_full = opticflow->av_dist.y_full*(n_new)/
            (opticflow->av_dist.y_full*opticflow->previous_tracked_cnt + (float)found_dist_sum.y_full);
        opticflow->width_dist_ratio *= width_corr_found.x_full;
        opticflow->height_dist_ratio *= width_corr_found.y_full;
        // For next loop and rest of the calculations, new corners now belong to the list
        result->tracked_cnt += opticflow->n_new_corners;
        opticflow->n_new_corners = 0;
      }
      // Find average position of corners
      struct point_tf av_corn_pos;
      av_corn_pos.x_full = 0, av_corn_pos.y_full = 0;
      for (uint16_t i = 0; i < result->tracked_cnt; i++) {
        av_corn_pos.x_full += opticflow->fast9_ret_corners[i].x_full;
        av_corn_pos.y_full += opticflow->fast9_ret_corners[i].y_full;
      }
      opticflow->cg_corners.x_full = av_corn_pos.x_full/result->tracked_cnt;
      opticflow->cg_corners.y_full = av_corn_pos.y_full/result->tracked_cnt;
      // Propagate ROI based on average position of corners
      opticflow->roi_center.x_full = (uint16_t)((int32_t)opticflow->cg_corners.x_full+opticflow->offset_roi_cg.x_full);
      opticflow->roi_center.y_full = (uint16_t)((int32_t)opticflow->cg_corners.y_full+opticflow->offset_roi_cg.y_full);
      opticflow->roi_center.x = opticflow->roi_center.x_full/opticflow->subpixel_factor;
      opticflow->roi_center.y = opticflow->roi_center.y_full/opticflow->subpixel_factor;
      // Find average distance to corner cg
      struct point_tf av_dist; av_dist.x_full = 0; av_dist.y_full = 0;
      for (uint16_t i = 0; i < result->tracked_cnt; i++) {
        av_dist.x_full += (uint32_t)abs((int32_t)opticflow->fast9_ret_corners[i].x_full-(int32_t)opticflow->cg_corners.x_full);
        av_dist.y_full += (uint32_t)abs((int32_t)opticflow->fast9_ret_corners[i].y_full-(int32_t)opticflow->cg_corners.y_full);
      }
      opticflow->av_dist.x_full = (float)av_dist.x_full/(float)result->tracked_cnt;
      opticflow->av_dist.y_full = (float)av_dist.y_full/(float)result->tracked_cnt;
      // Update ROI width based on new average distance
      opticflow->roiw = opticflow->width_dist_ratio*opticflow->av_dist.x_full/opticflow->subpixel_factor;
      opticflow->roih = opticflow->height_dist_ratio*opticflow->av_dist.y_full/opticflow->subpixel_factor;
      /*printf("New roi dimensions = (%f,%f)\n",opticflow->roiw,opticflow->roih);*/
      // Update ROI size based on the difference in movement between the left and right and top and bottom due to divergence
      // Divergence is defined opposite to the usual
      // roi[0].dx - roi[2].dx = (roi[0].x-roi[2].x)*div_size/subpixel = roiw*div_size/subpixel, this should logically be the increase in size
      // Update all roi features
      // Check which condition is most stringent: being within the window or the roi
      opticflow->roi[0] = (uint16_t)MAX(opticflow->min_dist_from_edge,((int16_t)opticflow->roi_center.x_full -
          (int16_t)(opticflow->roiw*opticflow->subpixel_factor/2))/(int16_t)opticflow->subpixel_factor);
      opticflow->roi[2] = (uint16_t)MIN((int16_t)opticflow->img_gray.w-opticflow->min_dist_from_edge,((int16_t)opticflow->roi_center.x_full +
          (int16_t)(opticflow->roiw*opticflow->subpixel_factor/2))/(int16_t)opticflow->subpixel_factor);
      opticflow->roi[1] = (uint16_t)MAX(opticflow->min_dist_from_edge,((int16_t)opticflow->roi_center.y_full -
          (int16_t)(opticflow->roih*opticflow->subpixel_factor/2))/(int16_t)opticflow->subpixel_factor);
      opticflow->roi[3] = (uint16_t)MIN((int16_t)opticflow->img_gray.h-opticflow->min_dist_from_edge,((int16_t)opticflow->roi_center.y_full +
          (int16_t)(opticflow->roih*opticflow->subpixel_factor/2))/(int16_t)opticflow->subpixel_factor);
      // Prepare for next iteration
      opticflow->previous_tracked_cnt = result->tracked_cnt;
      for (uint16_t i = 0; i < result->tracked_cnt; i++) {
        opticflow->previous_fast9_ret_corners[i].x_full = opticflow->fast9_ret_corners[i].x_full;
        opticflow->previous_fast9_ret_corners[i].y_full = opticflow->fast9_ret_corners[i].y_full;
        opticflow->previous_fast9_ret_corners[i].x = opticflow->fast9_ret_corners[i].x;
        opticflow->previous_fast9_ret_corners[i].y = opticflow->fast9_ret_corners[i].y;
      }
    }
    // Check if width takes up to big a part of the screen, if so reinitialize for the next frame
    if(opticflow->roiw > 0.75*opticflow->img_gray.w || opticflow->roih > 0.75*opticflow->img_gray.h){
      opticflow->object_tracking_set = true;
      opticflow->object_tracking_reset = true;
    }
    // Display ROI
    struct point_tf roi_to_show[4];
    roi_to_show[0].x = opticflow->roi[0];
    roi_to_show[0].y = opticflow->roi[1];
    roi_to_show[1].x = opticflow->roi[0];
    roi_to_show[1].y = opticflow->roi[3];
    roi_to_show[2].x = opticflow->roi[2];
    roi_to_show[2].y = opticflow->roi[1];
    roi_to_show[3].x = opticflow->roi[2];
    roi_to_show[3].y = opticflow->roi[3];
#if OPTICFLOW_SHOW_CORNERS
  static uint8_t white[4] = {127, 255, 127, 255};
  image_show_points_color(img, roi_to_show, 4, white);
#endif
  }


  // Determine quality of noise measurement for state filter
  //TODO develop a noise model based on groundtruth
  //result->noise_measurement = 1 - (float)result->tracked_cnt / ((float)opticflow->max_track_corners * 1.25f);
  result->noise_measurement = 0.25;

  free(vectors);
  image_switch(&opticflow->img_gray, &opticflow->prev_img_gray);

  return true;
}

/* manage_flow_features - Update list of corners to be tracked by LK
 * Remembers previous points and tries to find new points in less dense
 * areas of the image first.
 *
 */
static void manage_flow_features(struct opticflow_t *opticflow, struct opticflow_result_t *result, uint16_t *roi)
{
  // first check if corners have not moved too close together due to flow:
  int16_t c1 = 0;
  while (c1 < (int16_t)result->corner_cnt - 1) {
    bool exists = false;
    for (int16_t i = c1 + 1; i < result->corner_cnt; i++) {
      if (abs((int16_t)opticflow->fast9_ret_corners[c1].x - (int16_t)opticflow->fast9_ret_corners[i].x) <
          opticflow->fast9_min_distance / 2
          && abs((int16_t)opticflow->fast9_ret_corners[c1].y - (int16_t)opticflow->fast9_ret_corners[i].y) <
          opticflow->fast9_min_distance / 2) {
        // if too close, replace the corner with the last one in the list:
        opticflow->fast9_ret_corners[c1].x = opticflow->fast9_ret_corners[result->corner_cnt - 1].x;
        opticflow->fast9_ret_corners[c1].y = opticflow->fast9_ret_corners[result->corner_cnt - 1].y;
        opticflow->fast9_ret_corners[c1].count = opticflow->fast9_ret_corners[result->corner_cnt - 1].count;
        opticflow->fast9_ret_corners[c1].x_full = opticflow->fast9_ret_corners[result->corner_cnt - 1].x_full;
        opticflow->fast9_ret_corners[c1].y_full = opticflow->fast9_ret_corners[result->corner_cnt - 1].y_full;
        // decrease the number of corners:
        result->corner_cnt--;
        exists = true;
        // no further checking required for the removed corner
        break;
      }
    }
    // if the corner has been replaced, the new corner in position c1 has to be checked again:
    if (!exists) { c1++; }
  }

  // no need for "per region" re-detection when there are no previous corners
  if ((opticflow->object_tracking) && ((!opticflow->fast9_region_detect) || (result->corner_cnt == 0))) {
    fast9_detect(&opticflow->prev_img_gray, opticflow->fast9_threshold, opticflow->fast9_min_distance,
                opticflow->fast9_padding, opticflow->fast9_padding, &result->corner_cnt,
                &opticflow->fast9_rsize,
                &opticflow->fast9_ret_corners,
                roi);
  } else if ((!opticflow->fast9_region_detect) || (result->corner_cnt == 0)) {
    fast9_detect(&opticflow->prev_img_gray, opticflow->fast9_threshold, opticflow->fast9_min_distance,
                 opticflow->fast9_padding, opticflow->fast9_padding, &result->corner_cnt,
                 &opticflow->fast9_rsize,
                 &opticflow->fast9_ret_corners,
                 roi);

  } else {
    // allocating memory and initializing the 2d array that holds the number of corners per region and its index (for the sorting)
    uint16_t **region_count = calloc(opticflow->fast9_num_regions, sizeof(uint16_t *));
    for (uint16_t i = 0; i < opticflow->fast9_num_regions; i++) {
      region_count[i] = calloc(2, sizeof(uint16_t));
      region_count[i][0] = 0;
      region_count[i][1] = i;
    }

    uint16_t root_regions = (uint16_t)sqrtf((float)opticflow->fast9_num_regions);
    int region_index;
    for (uint16_t i = 0; i < result->corner_cnt; i++) {
      region_index = (opticflow->fast9_ret_corners[i].x * root_regions / opticflow->prev_img_gray.w
         + root_regions * (opticflow->fast9_ret_corners[i].y * root_regions / opticflow->prev_img_gray.h));
      region_index = (region_index < opticflow->fast9_num_regions) ? region_index : opticflow->fast9_num_regions - 1;
      region_count[region_index][0]++;
    }

    //sorting region_count array according to first column (number of corners).
    qsort(region_count, opticflow->fast9_num_regions, sizeof(region_count[0]), cmp_array);

    // Detecting corners from the region with the less to the one with the most, until a desired total is reached.
    for (uint16_t i = 0; i < opticflow->fast9_num_regions && result->corner_cnt < 2 * opticflow->max_track_corners; i++) {
      // Find the boundaries of the region of interest
      roi[0] = (region_count[i][1] % root_regions) * (opticflow->prev_img_gray.w / root_regions);
      roi[1] = (region_count[i][1] / root_regions) * (opticflow->prev_img_gray.h / root_regions);
      roi[2] = roi[0] + (opticflow->prev_img_gray.w / root_regions);
      roi[3] = roi[1] + (opticflow->prev_img_gray.h / root_regions);

      struct point_tf *new_corners = calloc(opticflow->fast9_rsize, sizeof(struct point_tf));
      uint16_t new_count = 0;

      fast9_detect(&opticflow->prev_img_gray, opticflow->fast9_threshold, opticflow->fast9_min_distance,
                   opticflow->fast9_padding, opticflow->fast9_padding, &new_count,
                   &opticflow->fast9_rsize, &new_corners, roi);

      // check that no identified points already exist in list
      for (uint16_t j = 0; j < new_count; j++) {
        bool exists = false;
        for (uint16_t k = 0; k < result->corner_cnt; k++) {
          if (abs((int16_t)new_corners[j].x - (int16_t)opticflow->fast9_ret_corners[k].x) < (int16_t)opticflow->fast9_min_distance
              && abs((int16_t)new_corners[j].y - (int16_t)opticflow->fast9_ret_corners[k].y) < (int16_t)
              opticflow->fast9_min_distance) {
            exists = true;
            break;
          }
        }
        if (!exists) {
          opticflow->fast9_ret_corners[result->corner_cnt].x = new_corners[j].x;
          opticflow->fast9_ret_corners[result->corner_cnt].y = new_corners[j].y;
          opticflow->fast9_ret_corners[result->corner_cnt].count = 0;
          opticflow->fast9_ret_corners[result->corner_cnt].x_full = new_corners[j].x * opticflow->subpixel_factor;
          opticflow->fast9_ret_corners[result->corner_cnt].y_full = new_corners[j].y * opticflow->subpixel_factor;
          result->corner_cnt++;

          if (result->corner_cnt >= opticflow->fast9_rsize) {
            break;
          }
        }
      }

      free(new_corners);
    }
    for (uint16_t i = 0; i < opticflow->fast9_num_regions; i++) {
      free(region_count[i]);
    }
  }
}

/**
 * Run the optical flow with EDGEFLOW on a new image frame
 * @param[in] *opticflow The opticalflow structure that keeps track of previous images
 * @param[in] *state The state of the drone
 * @param[in] *img The image frame to calculate the optical flow from
 * @param[out] *result The optical flow result
 * @param computation successful
 */
bool calc_edgeflow_tot(struct opticflow_t *opticflow, struct image_t *img,
                       struct opticflow_result_t *result)
{
  // Define Static Variables
  static struct edge_hist_t edge_hist[MAX_HORIZON];
  static uint8_t current_frame_nr = 0;
  struct edge_flow_t edgeflow;
  static uint8_t previous_frame_offset[2] = {1, 1};

  // Define Normal variables
  struct edgeflow_displacement_t displacement;
  displacement.x = calloc(img->w, sizeof(int32_t));
  displacement.y = calloc(img->h, sizeof(int32_t));

  // If the methods just switched to this one, reintialize the
  // array of edge_hist structure.
  if (opticflow->just_switched_method == 1 && edge_hist[0].x == NULL) {
    int i;
    for (i = 0; i < MAX_HORIZON; i++) {
      edge_hist[i].x = calloc(img->w, sizeof(int32_t));
      edge_hist[i].y = calloc(img->h, sizeof(int32_t));
      FLOAT_EULERS_ZERO(edge_hist[i].eulers);
    }
  }

  uint16_t disp_range;
  if (opticflow->search_distance < DISP_RANGE_MAX) {
    disp_range = opticflow->search_distance;
  } else {
    disp_range = DISP_RANGE_MAX;
  }

  uint16_t window_size;

  if (opticflow->window_size < MAX_WINDOW_SIZE) {
    window_size = opticflow->window_size;
  } else {
    window_size = MAX_WINDOW_SIZE;
  }

  uint16_t RES = opticflow->resolution_factor;

  //......................Calculating EdgeFlow..................... //

  // Calculate current frame's edge histogram
  int32_t *edge_hist_x = edge_hist[current_frame_nr].x;
  int32_t *edge_hist_y = edge_hist[current_frame_nr].y;
  calculate_edge_histogram(img, edge_hist_x, 'x', 0);
  calculate_edge_histogram(img, edge_hist_y, 'y', 0);


  // Copy frame time and angles of image to calculated edge histogram
  edge_hist[current_frame_nr].frame_time = img->ts;
  edge_hist[current_frame_nr].eulers = img->eulers;

  // Calculate which previous edge_hist to compare with the current
  uint8_t previous_frame_nr[2];
  calc_previous_frame_nr(result, opticflow, current_frame_nr, previous_frame_offset, previous_frame_nr);

  //Select edge histogram from the previous frame nr
  int32_t *prev_edge_histogram_x = edge_hist[previous_frame_nr[0]].x;
  int32_t *prev_edge_histogram_y = edge_hist[previous_frame_nr[1]].y;

  //Calculate the corresponding derotation of the two frames
  int16_t der_shift_x = 0;
  int16_t der_shift_y = 0;

  if (opticflow->derotation) {
    der_shift_x = (int16_t)((edge_hist[current_frame_nr].eulers.phi - edge_hist[previous_frame_nr[0]].eulers.phi) *
                            OPTICFLOW_FX * opticflow->derotation_correction_factor_x);
    der_shift_y = (int16_t)((edge_hist[current_frame_nr].eulers.theta - edge_hist[previous_frame_nr[1]].eulers.theta) *
                            OPTICFLOW_FY * opticflow->derotation_correction_factor_y);
  }

  // Estimate pixel wise displacement of the edge histograms for x and y direction
  calculate_edge_displacement(edge_hist_x, prev_edge_histogram_x,
                              displacement.x, img->w,
                              window_size, disp_range,  der_shift_x);
  calculate_edge_displacement(edge_hist_y, prev_edge_histogram_y,
                              displacement.y, img->h,
                              window_size, disp_range, der_shift_y);

  // Fit a line on the pixel displacement to estimate
  // the global pixel flow and divergence (RES is resolution)
  line_fit(displacement.x, &edgeflow.div_x,
           &edgeflow.flow_x, img->w,
           window_size + disp_range, RES);
  line_fit(displacement.y, &edgeflow.div_y,
           &edgeflow.flow_y, img->h,
           window_size + disp_range, RES);

  /* Save Resulting flow in results
   * Warning: The flow detected here is different in sign
   * and size, therefore this will be divided with
   * the same subpixel factor and multiplied by -1 to make it
   * on par with the LK algorithm in opticalflow_calculator.c
   * */
  edgeflow.flow_x = -1 * edgeflow.flow_x;
  edgeflow.flow_y = -1 * edgeflow.flow_y;

  edgeflow.flow_x = (int16_t)edgeflow.flow_x / previous_frame_offset[0];
  edgeflow.flow_y = (int16_t)edgeflow.flow_y / previous_frame_offset[1];

  result->flow_x = (int16_t)edgeflow.flow_x / RES;
  result->flow_y = (int16_t)edgeflow.flow_y / RES;

  //Fill up the results optic flow to be on par with LK_fast9
  result->flow_der_x =  result->flow_x;
  result->flow_der_y =  result->flow_y;
  result->corner_cnt = getAmountPeaks(edge_hist_x, 500 , img->w);
  result->tracked_cnt = getAmountPeaks(edge_hist_x, 500 , img->w);
  result->divergence = -1.0 * (float)edgeflow.div_x /
                       RES; // Also multiply the divergence with -1.0 to make it on par with the LK algorithm of
  result->div_size =
    result->divergence;  // Fill the div_size with the divergence to atleast get some divergenge measurement when switching from LK to EF
  result->surface_roughness = 0.0f;

  //......................Calculating VELOCITY ..................... //

  /*Estimate fps per direction
   * This is the fps with adaptive horizon for subpixel flow, which is not similar
   * to the loop speed of the algorithm. The faster the quadcopter flies
   * the higher it becomes
  */
  float fps_x = 0;
  float fps_y = 0;
  float time_diff_x = (float)(timeval_diff(&edge_hist[previous_frame_nr[0]].frame_time, &img->ts)) / 1000.;
  float time_diff_y = (float)(timeval_diff(&edge_hist[previous_frame_nr[1]].frame_time, &img->ts)) / 1000.;
  fps_x = 1 / (time_diff_x);
  fps_y = 1 / (time_diff_y);

  result->fps = fps_x;

  // TODO scale flow to rad/s here

  // Calculate velocity
  result->vel_cam.x = edgeflow.flow_x * fps_x * agl_dist_value_filtered * OPTICFLOW_FX / RES;
  result->vel_cam.y = edgeflow.flow_y * fps_y * agl_dist_value_filtered * OPTICFLOW_FY / RES;
  result->vel_cam.z = result->divergence * fps_x * agl_dist_value_filtered;

  //Apply a  median filter to the velocity if wanted
  if (opticflow->median_filter == true) {
    UpdateMedianFilterVect3Float(vel_filt, result->vel_cam);
  }

  result->noise_measurement = 0.2;

#if OPTICFLOW_SHOW_FLOW
  draw_edgeflow_img(img, edgeflow, prev_edge_histogram_x, edge_hist_x);
#endif
  // Increment and wrap current time frame
  current_frame_nr = (current_frame_nr + 1) % MAX_HORIZON;

  // Free alloc'd variables
  free(displacement.x);
  free(displacement.y);

  return true;
}


/**
 * Run the optical flow on a new image frame
 * @param[in] *opticflow The opticalflow structure that keeps track of previous images
 * @param[in] *state The state of the drone
 * @param[in] *img The image frame to calculate the optical flow from
 * @param[out] *result The optical flow result
 */
bool opticflow_calc_frame(struct opticflow_t *opticflow, struct image_t *img,
                          struct opticflow_result_t *result)
{
  bool flow_successful = false;
  // A switch counter that checks in the loop if the current method is similar,
  // to the previous (for reinitializing structs)
  static int8_t switch_counter = -1;
  if (switch_counter != opticflow->method) {
    opticflow->just_switched_method = true;
    switch_counter = opticflow->method;
    // Clear the static result
    memset(result, 0, sizeof(struct opticflow_result_t));
  } else {
    opticflow->just_switched_method = false;
  }

  // Switch between methods (0 = fast9/lukas-kanade, 1 = EdgeFlow)
  if (opticflow->method == 0) {
    flow_successful = calc_fast9_lukas_kanade(opticflow, img, result);
  } else if (opticflow->method == 1) {
    flow_successful = calc_edgeflow_tot(opticflow, img, result);
  }

  /* Rotate velocities from camera frame coordinates to body coordinates for control
  * IMPORTANT!!! This frame to body orientation should be the case for the Parrot
  * ARdrone and Bebop, however this can be different for other quadcopters
  * ALWAYS double check!
  */
  float_rmat_transp_vmult(&result->vel_body, &body_to_cam, &result->vel_cam);

  return flow_successful;
}

/**
 * Calculate the difference from start till finish
 * @param[in] *starttime The start time to calculate the difference from
 * @param[in] *finishtime The finish time to calculate the difference from
 * @return The difference in milliseconds
 */
static uint32_t timeval_diff(struct timeval *starttime, struct timeval *finishtime)
{
  uint32_t msec;
  msec = (finishtime->tv_sec - starttime->tv_sec) * 1000;
  msec += (finishtime->tv_usec - starttime->tv_usec) / 1000;
  return msec;
}

/**
 * Compare two flow vectors based on flow distance
 * Used for sorting.
 * @param[in] *a The first flow vector (should be vect flow_t)
 * @param[in] *b The second flow vector (should be vect flow_t)
 * @return Negative if b has more flow than a, 0 if the same and positive if a has more flow than b
 */
static int cmp_flow(const void *a, const void *b)
{
  const struct flow_t *a_p = (const struct flow_t *)a;
  const struct flow_t *b_p = (const struct flow_t *)b;
  return (a_p->flow_x * a_p->flow_x + a_p->flow_y * a_p->flow_y) - (b_p->flow_x * b_p->flow_x + b_p->flow_y *
         b_p->flow_y);
}

/**
 * Compare the rows of an integer (uint16_t) 2D array based on the first column.
 * Used for sorting.
 * @param[in] *a The first row (should be *uint16_t)
 * @param[in] *b The second flow vector (should be *uint16_t)
 * @return Negative if a[0] < b[0],0 if a[0] == b[0] and positive if a[0] > b[0]
 */
static int cmp_array(const void *a, const void *b)
{
  uint16_t *pa = *(uint16_t **)a;
  uint16_t *pb = *(uint16_t **)b;
  return pa[0] - pb[0];
}

void check_for_features(struct opticflow_t *opticflow, struct opticflow_result_t *result)
{
  // Check if corners haven't moved too close to each other
  int16_t c1 = 0;
  while (c1 < (int16_t)result->corner_cnt - 1) {
    bool exists = false;
    for (int16_t i = c1 + 1; i < result->corner_cnt; i++) {
      if (abs((int16_t)opticflow->fast9_ret_corners[c1].x - (int16_t)opticflow->fast9_ret_corners[i].x) <
          opticflow->fast9_min_distance / 2
          && abs((int16_t)opticflow->fast9_ret_corners[c1].y - (int16_t)opticflow->fast9_ret_corners[i].y) <
          opticflow->fast9_min_distance / 2) {
        // if too close, replace the corner with the last one in the list:
        opticflow->fast9_ret_corners[c1].x = opticflow->fast9_ret_corners[result->corner_cnt - 1].x;
        opticflow->fast9_ret_corners[c1].y = opticflow->fast9_ret_corners[result->corner_cnt - 1].y;
        opticflow->fast9_ret_corners[c1].count = opticflow->fast9_ret_corners[result->corner_cnt - 1].count;
        opticflow->fast9_ret_corners[c1].x_full = opticflow->fast9_ret_corners[result->corner_cnt - 1].x_full;
        opticflow->fast9_ret_corners[c1].y_full = opticflow->fast9_ret_corners[result->corner_cnt - 1].y_full;
        // decrease the number of corners:
        result->corner_cnt--;
        exists = true;
        // no further checking required for the removed corner
        break;
      }
    }
    // if the corner has been replaced, the new corner in position c1 has to be checked again:
    if (!exists) { c1++; }
  }
  // Look for new corners
  // Define area to look in: should be in roi, but also somewhat centered in the image, since we don't
  // want to lose it again right away
  uint16_t new_corner_roi[4];
  opticflow->new_corners_found = 0;
  // Check which condition is most stringent
  new_corner_roi[0] = (uint16_t)MAX(opticflow->min_dist_from_edge,(int16_t)opticflow->roi[0]);
  new_corner_roi[2] = (uint16_t)MIN((int16_t)opticflow->img_gray.w-opticflow->min_dist_from_edge,(int16_t)opticflow->roi[2]);
  new_corner_roi[1] = (uint16_t)MAX(opticflow->min_dist_from_edge,(int16_t)opticflow->roi[1]);
  new_corner_roi[3] = (uint16_t)MIN((int16_t)opticflow->img_gray.h-opticflow->min_dist_from_edge,(int16_t)opticflow->roi[3]);
  fast9_detect(&opticflow->img_gray, opticflow->fast9_threshold,opticflow->fast9_min_distance,
       opticflow->fast9_padding, opticflow->fast9_padding, &opticflow->new_corners_found,
       &opticflow->nr_of_new_corners_to_detect,
       &opticflow->new_fast9_ret_corners,
       new_corner_roi);
  // Check which new corners are far away enough from the other corners to be selected
  uint8_t non_double_corners = 0;
  uint8_t index = 0;
  // Fill random index array, so we don't only get corners close together
  uint32_t rand_arr[opticflow->new_corners_found];
  for (uint32_t i=0;i<opticflow->new_corners_found;i++){
    rand_arr[i] = rand() % opticflow->new_corners_found;
    if(i>0){
      // Check that element is not already present in array
      for (uint32_t j=0;j<i;j++){
        if(rand_arr[i]==rand_arr[j]){
          i--;
          break;
        }
      }
    }
  }
  uint8_t index_array[opticflow->n_new_corners];
  // Check if while loop has to be broken, due to a lack of corners
  opticflow->insufficient_corners = false;
  while (non_double_corners < opticflow->n_new_corners){
    // Select element in new_corners
    bool exists = false;
    uint32_t index_rand = rand_arr[index];
    // Compare with every element in current array
    for (uint8_t i=0;i<result->corner_cnt;i++){
      // If distance too small break for loop and go to next element
      if (abs((int16_t)opticflow->fast9_ret_corners[i].x - (int16_t)opticflow->new_fast9_ret_corners[index_rand].x) <
                opticflow->fast9_min_distance / 2
                && abs((int16_t)opticflow->fast9_ret_corners[i].y - (int16_t)opticflow->new_fast9_ret_corners[index_rand].y) <
                opticflow->fast9_min_distance / 2) {
        exists = true;
        break;
      }
    }
    // At end of for loop, not broken: add index to index_list
    if (!exists){
      index_array[non_double_corners]= index_rand;
      non_double_corners++;
    }
    index++;
    // If not enough non duplicate corners can be found
    /*if (index >= opticflow->nr_of_new_corners_to_detect){*/
    if (index >= opticflow->new_corners_found){
      opticflow->n_new_corners = non_double_corners;
      printf("Trying the break\n");
      opticflow->insufficient_corners = true;
      break;
    }
  }
  if(opticflow->insufficient_corners){
    opticflow->broken_in_row++;
    if(opticflow->broken_in_row>1){
      printf("Final landing forced, due to lack of corners\n");
      opticflow->landing = true;
    }
  } else{
    opticflow->broken_in_row = 0;
  }
  // Add them at the end of the corner array
  for (uint8_t i=0;i<opticflow->n_new_corners;i++){
    opticflow->fast9_ret_corners[result->corner_cnt+i].x = opticflow->new_fast9_ret_corners[index_array[i]].x;
    opticflow->fast9_ret_corners[result->corner_cnt+i].y = opticflow->new_fast9_ret_corners[index_array[i]].y;
    opticflow->fast9_ret_corners[result->corner_cnt+i].x_full = opticflow->new_fast9_ret_corners[index_array[i]].x*opticflow->subpixel_factor;
    opticflow->fast9_ret_corners[result->corner_cnt+i].y_full = opticflow->new_fast9_ret_corners[index_array[i]].y*opticflow->subpixel_factor;
  }
}

// Initialize object to be tracked
void init_object_tracking(struct opticflow_t *opticflow){
  /*opticflow->nr_of_object_corners = NR_OF_CORNERS_TO_TRACK;*/
  //opticflow->max_track_corners = NR_OF_CORNERS_TO_TRACK;
  opticflow->object_tracking_set = false;

  opticflow->actfast_long_step = OPTICFLOW_ACTFAST_LONG_STEP_OBJECT;
  opticflow->actfast_short_step = OPTICFLOW_ACTFAST_SHORT_STEP_OBJECT;

  // Define region of interest at initialization
  opticflow->offset_defined = false;
  if(!opticflow->object_tracking_reset){
    opticflow->roi_center.x = opticflow->roi_center_init.x;
    opticflow->roi_center.y = opticflow->roi_center_init.y;
    opticflow->roih = opticflow->roih_init;
    opticflow->roiw = opticflow->roiw_init;
    struct EnuCoor_f *enu = stateGetPositionEnu_f();
    struct FloatEulers *eulers = stateGetNedToBodyEulers_f();
    opticflow->start_pos.x = enu->x;opticflow->start_pos.y = enu->y;opticflow->start_pos.z = enu->z;
    opticflow->start_angles.phi = eulers->phi;
    opticflow->start_angles.theta = eulers->theta;
    opticflow->start_angles.psi = eulers->psi;
  }
  else{
    opticflow->t_new_window[opticflow->n_reinits] = get_sys_time_float();
    opticflow->n_reinits++;
    opticflow->roih = 0.5*opticflow->img_gray.h;
    opticflow->roiw = 0.5*opticflow->img_gray.w;
  }
  opticflow->ibvs_init = true;

  opticflow->min_dist_from_edge = 10;

  opticflow->roi_center.x_full = opticflow->roi_center.x*opticflow->subpixel_factor;
  opticflow->roi_center.y_full = opticflow->roi_center.y*opticflow->subpixel_factor;

  opticflow->roi[0] = opticflow->roi_center.x-(uint16_t)opticflow->roiw/2;
  opticflow->roi[1] = opticflow->roi_center.y-(uint16_t)opticflow->roih/2;
  opticflow->roi[2] = opticflow->roi_center.x+(uint16_t)opticflow->roiw/2;
  opticflow->roi[3] = opticflow->roi_center.y+(uint16_t)opticflow->roih/2;

  opticflow->roi_full[0] = opticflow->roi[0]*opticflow->subpixel_factor;
  opticflow->roi_full[1] = opticflow->roi[1]*opticflow->subpixel_factor;
  opticflow->roi_full[2] = opticflow->roi[2]*opticflow->subpixel_factor;
  opticflow->roi_full[3] = opticflow->roi[3]*opticflow->subpixel_factor;

}
