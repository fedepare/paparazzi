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
#include "lib/vision/undistortion.h"
#include "size_divergence.h"
#include "linear_flow_fit.h"
#include "modules/sonar/agl_dist.h"

// to get the definition of front_camera / bottom_camera
#include BOARD_CONFIG

// whether to show the flow and corners:
#define OPTICFLOW_SHOW_CORNERS 1

#define EXHAUSTIVE_FAST 0
#define ACT_FAST 1
// TODO: these are now adapted, but perhaps later could be a setting:
uint16_t n_time_steps = 10;

// What methods are run to determine divergence, lateral flow, etc.
// SIZE_DIV looks at line sizes and only calculates divergence
#define SIZE_DIV 1
// LINEAR_FIT makes a linear optical flow field fit and extracts a lot of information:
// relative velocities in x, y, z (divergence / time to contact), the slope of the surface, and the surface roughness.
#define LINEAR_FIT 1

#ifndef OPTICFLOW_CORNER_METHOD
#define OPTICFLOW_CORNER_METHOD EXHAUSTIVE_FAST
#endif
PRINT_CONFIG_VAR(OPTICFLOW_CORNER_METHOD)

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
#define OPTICFLOW_FAST9_ADAPTIVE FALSE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_FAST9_ADAPTIVE)

#ifndef OPTICFLOW_FAST9_THRESHOLD
#define OPTICFLOW_FAST9_THRESHOLD 35
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
#define OPTICFLOW_FAST9_REGION_DETECT 0
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

// Tracking back flow to make the accepted flow vectors more robust:
// Default is false, as it does take extra processing time
#ifndef OPTICFLOW_TRACK_BACK
#define OPTICFLOW_TRACK_BACK TRUE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_TRACK_BACK)

// Whether to draw the flow on the image:
// False by default, since it changes the image and costs time.
#ifndef OPTICFLOW_SHOW_FLOW
#define OPTICFLOW_SHOW_FLOW FALSE
#endif
PRINT_CONFIG_VAR(OPTICFLOW_SHOW_FLOW)



//Include median filter
#include "filters/median_filter.h"
struct MedianFilter3Float vel_filt;
struct FloatRMat body_to_cam;

/* Functions only used here */
static uint32_t timeval_diff(struct timeval *starttime, struct timeval *finishtime);
static int cmp_flow(const void *a, const void *b);
static int cmp_array(const void *a, const void *b);
static void manage_flow_features(struct opticflow_t *opticflow, uint16_t *roi);
static void init_object_tracking(struct opticflow_t *opticflow, struct object_tracker_t *tracker);

static void remove_bad_vectors(struct opticflow_t *opticflow, struct flow_t *vectors);
static void check_tracked_points_corners(struct opticflow_t *opticflow, struct flow_t *vectors);
static void check_back_flow(struct opticflow_t *opticflow, struct flow_t *vecotrs);
static struct flow_t *predict_flow_vectors(struct flow_t *flow_vectors, uint16_t n_points, float phi_diff,
    float theta_diff, float psi_diff, struct opticflow_t *opticflow);
static void compute_global_flow(struct flow_t *vectors, struct opticflow_t *opticflow, struct opticflow_result_t *result);
static void update_object_roi(struct opticflow_t *opticflow, struct image_t *img, struct opticflow_result_t *result, struct object_tracker_t *tracker);

struct object_tracker_t tracker_glob;
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
  opticflow->track_back = OPTICFLOW_TRACK_BACK;
  opticflow->show_flow = OPTICFLOW_SHOW_FLOW;
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
  opticflow->fast9_ret_corners = calloc(opticflow->fast9_rsize, sizeof(struct point_t));
  opticflow->fast9_rsize_prev = opticflow->fast9_rsize;
  opticflow->fast9_ret_corners_prev = calloc(opticflow->fast9_rsize_prev, sizeof(struct point_t));

  opticflow->corner_method = OPTICFLOW_CORNER_METHOD;
  opticflow->actfast_long_step = OPTICFLOW_ACTFAST_LONG_STEP;
  opticflow->actfast_short_step = OPTICFLOW_ACTFAST_SHORT_STEP;
  opticflow->actfast_min_gradient = OPTICFLOW_ACTFAST_MIN_GRADIENT;
  opticflow->actfast_gradient_method = OPTICFLOW_ACTFAST_GRADIENT_METHOD;

  struct FloatEulers euler = {OPTICFLOW_BODY_TO_CAM_PHI, OPTICFLOW_BODY_TO_CAM_THETA, OPTICFLOW_BODY_TO_CAM_PSI};
  float_rmat_of_eulers(&body_to_cam, &euler);

  init_object_tracking(opticflow, &tracker_glob);
}

#include "pprzlink/dl_protocol.h"
void update_roi_dl(uint8_t *buf)
{
  tracker_glob.roi_centriod_x = DL_VIDEO_ROI_startx(buf) + DL_VIDEO_ROI_width(buf)/2;
  tracker_glob.roi_centriod_y = DL_VIDEO_ROI_starty(buf) + DL_VIDEO_ROI_height(buf)/2;
  tracker_glob.roi_w = DL_VIDEO_ROI_width(buf);
  tracker_glob.roi_h = DL_VIDEO_ROI_height(buf);

  tracker_glob.roi[0] = (uint16_t)Min(OPTICFLOW_CAMERA.output_size.w-1, Max(0.f, tracker_glob.roi_centriod_x - tracker_glob.roi_w/2));
  tracker_glob.roi[1] = (uint16_t)Min(OPTICFLOW_CAMERA.output_size.h-1, Max(0.f, tracker_glob.roi_centriod_y - tracker_glob.roi_h/2));
  tracker_glob.roi[2] = (uint16_t)Max(0.f, Min(OPTICFLOW_CAMERA.output_size.w-1, tracker_glob.roi_centriod_x + tracker_glob.roi_w/2));
  tracker_glob.roi[3] = (uint16_t)Max(0.f, Min(OPTICFLOW_CAMERA.output_size.h-1, tracker_glob.roi_centriod_y + tracker_glob.roi_h/2));

  tracker_glob.roi_defined = true;
}

/**
 * Run the optical flow on a new image frame
 * @param[in] *opticflow The opticalflow structure that keeps track of previous images
 * @param[in] *state The state of the drone
 * @param[in] *img The image frame to calculate the optical flow from
 * @param[out] *result The optical flow result
 */
#include "mcu_periph/sys_time.h"
bool opticflow_calc_frame(struct opticflow_t *opticflow, struct image_t *img, struct opticflow_result_t *result)
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
    //float start = get_sys_time_float();
    flow_successful = calc_fast9_lukas_kanade(opticflow, img, result, tracker_glob.roi);
    //printf("time: %f ms\n", (get_sys_time_float() - start) * 1000.f);
    update_object_roi(opticflow, img, result, &tracker_glob);
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
 * Run the optical flow with fast9 and lukaskanade on a new image frame
 * @param[in] *opticflow The opticalflow structure that keeps track of previous images
 * @param[in] *state The state of the drone
 * @param[in] *img The image frame to calculate the optical flow from
 * @param[out] *result The optical flow result
 * @return Was optical flow successful
 */
bool calc_fast9_lukas_kanade(struct opticflow_t *opticflow, struct image_t *img,
                             struct opticflow_result_t *result, uint16_t *roi)
{
  if (roi){
    if(roi[0] >= img->w) { roi[0] = 0; }
    if(roi[1] >= img->h) { roi[1] = 0; }
    if(roi[2] >= img->w) { roi[2] = img->w - 1; }
    if(roi[3] >= img->h) { roi[3] = img->h - 1; }
  }

  if (opticflow->just_switched_method) {
    // Create the image buffers
    image_create(&opticflow->img_gray, img->w, img->h, IMAGE_GRAYSCALE);
    image_create(&opticflow->prev_img_gray, img->w, img->h, IMAGE_GRAYSCALE);

    // Set the previous values
    opticflow->got_first_img = false;

    // Init median filters with zeros
    InitMedianFilterVect3Float(vel_filt, MEDIAN_DEFAULT_SIZE);

    opticflow->corner_cnt = 0;
    opticflow->corner_cnt_prev = 0;
    opticflow->tracked_cnt = 0;
  }

  // Convert image to grayscale
  image_to_grayscale(img, &opticflow->img_gray);

  // Copy to previous image if not set
  if (!opticflow->got_first_img) {
    image_copy(&opticflow->img_gray, &opticflow->prev_img_gray);
    opticflow->got_first_img = true;
    return false;
  }

  // Update FPS for information
  float dt = timeval_diff(&(opticflow->prev_img_gray.ts), &(img->ts)) / 1000.f;
  if (dt > 1e-5) {
    result->fps = 1.f / dt;
  } else {
    return false;
  }

  // *************************************************************************************
  // Corner detection
  // *************************************************************************************
  if (opticflow->feature_management) {
    // if feature_management is selected and tracked corners drop below a threshold, redetect
    manage_flow_features(opticflow, roi);
  } else {
    // needs to be set to 0 because result is now static
    opticflow->corner_cnt = 0;

    if (opticflow->corner_method == EXHAUSTIVE_FAST) {
      // FAST corner detection
      // TODO: There is something wrong with fast9_detect destabilizing FPS. This problem is reduced with putting min_distance
      // to 0 (see defines), however a more permanent solution should be considered
      fast9_detect(&opticflow->prev_img_gray, opticflow->fast9_threshold, opticflow->fast9_min_distance,
                   opticflow->fast9_padding, opticflow->fast9_padding, &opticflow->corner_cnt,
                   &opticflow->fast9_rsize, &opticflow->fast9_ret_corners, roi);
    } else if (opticflow->corner_method == ACT_FAST) {
      // ACT-FAST corner detection:
      act_fast(&opticflow->prev_img_gray, opticflow->fast9_threshold, &opticflow->corner_cnt,
               &opticflow->fast9_ret_corners, opticflow->max_track_corners, n_time_steps,
               opticflow->actfast_long_step, opticflow->actfast_short_step, opticflow->actfast_min_gradient,
               opticflow->actfast_gradient_method);
    }

    // Adaptive threshold
    if (opticflow->fast9_adaptive) {
      // This works well for exhaustive FAST, but drives the threshold to the minimum for ACT-FAST:
      // Decrease and increase the threshold based on previous values
      if (opticflow->corner_cnt < 40) { // TODO: Replace 40 with OPTICFLOW_MAX_TRACK_CORNERS / 2
        // make detections easier:
        if (opticflow->fast9_threshold > FAST9_LOW_THRESHOLD) {
          opticflow->fast9_threshold--;
        }
        if (opticflow->corner_method == ACT_FAST) {
          n_time_steps++;
        }
      } else if (opticflow->corner_cnt > opticflow->max_track_corners * 2 && opticflow->fast9_threshold < FAST9_HIGH_THRESHOLD) {
        opticflow->fast9_threshold++;
        if (opticflow->corner_method == ACT_FAST && n_time_steps > 5) {
          n_time_steps--;
        }
      }
    }
  }

  // Check if we found some corners to track
  if (opticflow->corner_cnt < 3) {
    // Clear the result otherwise the previous values will be returned for this frame too
    VECT3_ASSIGN(result->vel_cam, 0, 0, 0);
    VECT3_ASSIGN(result->vel_body, 0, 0, 0);
    result->div_size = 0; result->divergence = 0;
    result->noise_measurement = 5.0;

    image_switch(&opticflow->img_gray, &opticflow->prev_img_gray);
    return false;
  }

#if OPTICFLOW_SHOW_CORNERS
  image_show_points(img, opticflow->fast9_ret_corners, opticflow->corner_cnt);
#endif

  // *************************************************************************************
  // Corner Tracking
  // *************************************************************************************
  // Execute a Lucas Kanade optical flow
  struct flow_t *vectors = NULL;
  if (opticflow->feature_management) {
    struct flow_t *vectors_prev = NULL;
    struct flow_t *vectors_new = NULL;
    // first only track corners that have been previously tracked
    // we split this such that newly tracked points are evenly distributed in the image
    opticflow->tracked_cnt = opticflow->corner_cnt_prev;
    if(opticflow->tracked_cnt){
      vectors_prev = opticFlowLK(&opticflow->img_gray, &opticflow->prev_img_gray, opticflow->fast9_ret_corners,
                                         &opticflow->tracked_cnt, opticflow->window_size / 2, opticflow->subpixel_factor,
                                         opticflow->max_iterations, opticflow->threshold_vec, opticflow->corner_cnt_prev,
                                         opticflow->pyramid_level, true);
    }
    // Track new corners
    opticflow->tracked_cnt = opticflow->corner_cnt - opticflow->corner_cnt_prev;
    if(opticflow->tracked_cnt){
      vectors_new = opticFlowLK(&opticflow->img_gray, &opticflow->prev_img_gray, &(opticflow->fast9_ret_corners[opticflow->corner_cnt_prev]),
                                         &opticflow->tracked_cnt, opticflow->window_size / 2, opticflow->subpixel_factor,
                                         opticflow->max_iterations, opticflow->threshold_vec, Max(0,(int16_t)opticflow->max_track_corners - opticflow->corner_cnt_prev),
                                         opticflow->pyramid_level, true);
    }

    vectors = malloc((opticflow->tracked_cnt + opticflow->corner_cnt_prev) * sizeof(struct flow_t));

    memcpy(vectors, vectors_prev, opticflow->corner_cnt_prev * sizeof(struct flow_t));
    memcpy(&(vectors[opticflow->corner_cnt_prev]), vectors_new, opticflow->tracked_cnt * sizeof(struct flow_t));
    opticflow->tracked_cnt += opticflow->corner_cnt_prev;

    free(vectors_prev);
    free(vectors_new);
  } else {
    opticflow->tracked_cnt = opticflow->corner_cnt;
    vectors = opticFlowLK(&opticflow->img_gray, &opticflow->prev_img_gray, opticflow->fast9_ret_corners,
                                       &opticflow->tracked_cnt, opticflow->window_size / 2, opticflow->subpixel_factor,
                                       opticflow->max_iterations, opticflow->threshold_vec, opticflow->max_track_corners,
                                       opticflow->pyramid_level, true);
  }

  // cleanup bad corners
  if (opticflow->track_back) {
    check_back_flow(opticflow, vectors);
  }
  if (opticflow->feature_management) {
    check_tracked_points_corners(opticflow, vectors);
  }

  if (opticflow->show_flow) {
    uint8_t color[4] = {127, 127, 127, 127};
    uint8_t bad_color[4] = {127, 0, 127, 0};
    image_show_flow_color(img, vectors, opticflow->tracked_cnt, opticflow->subpixel_factor, color, bad_color);
  }

  // *************************************************************************************
  // Next Loop Preparation and corner cleanup
  // *************************************************************************************
  remove_bad_vectors(opticflow, vectors);

  if (opticflow->tracked_cnt < 3) {
    // We got no flow
    result->flow_x = 0;
    result->flow_y = 0;

    free(vectors);
    image_switch(&opticflow->img_gray, &opticflow->prev_img_gray);
    return false;
  }

  // Global flow calculations
  static int n_samples = 100;
  // Estimate size divergence:
  if (SIZE_DIV) {
    result->div_size = get_size_divergence(vectors, opticflow->tracked_cnt, n_samples);// * result->fps;
  } else {
    result->div_size = 0.0f;
  }

  if (LINEAR_FIT) {
    // Linear flow fit (normally derotation should be performed before):
    static float error_threshold = 10.0f;
    static int n_iterations_RANSAC = 20;
    static int n_samples_RANSAC = 5;
    struct linear_flow_fit_info fit_info;
    int success_fit = analyze_linear_flow_field(vectors, opticflow->tracked_cnt,
        error_threshold, n_iterations_RANSAC, n_samples_RANSAC, img->w, img->h, &fit_info);

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

  // compute flow from vectors
  compute_global_flow(vectors, opticflow, result);
  // TODO scale flow to rad/s here

  // ***************
  // Flow Derotation
  // ***************
  float diff_flow_x = 0.f;
  float diff_flow_y = 0.f;

  if (opticflow->derotation) {
    float rotation_threshold = M_PI / 180.0f;
    if (fabs(opticflow->img_gray.eulers.phi - opticflow->prev_img_gray.eulers.phi) > rotation_threshold
        || fabs(opticflow->img_gray.eulers.theta - opticflow->prev_img_gray.eulers.theta) > rotation_threshold) {

      // do not apply the derotation if the rotation rates are too high:
      result->flow_der_x = 0.0f;
      result->flow_der_y = 0.0f;
    } else {
      // determine the roll, pitch, yaw differencces between the images.
      float phi_diff = opticflow->img_gray.eulers.phi - opticflow->prev_img_gray.eulers.phi;
      float theta_diff = opticflow->img_gray.eulers.theta - opticflow->prev_img_gray.eulers.theta;
      float psi_diff = opticflow->img_gray.eulers.psi - opticflow->prev_img_gray.eulers.psi;

      if (strcmp(OPTICFLOW_CAMERA.dev_name, front_camera.dev_name) == 0) {
        // frontal cam, predict individual flow vectors:
        struct flow_t *predicted_flow_vectors = predict_flow_vectors(vectors, opticflow->tracked_cnt, phi_diff, theta_diff,
                                                psi_diff, opticflow);
        if (opticflow->show_flow) {
          uint8_t color[4] = {255, 255, 255, 255};
          uint8_t bad_color[4] = {255, 255, 255, 255};
          image_show_flow_color(img, predicted_flow_vectors, opticflow->tracked_cnt, opticflow->subpixel_factor, color, bad_color);
        }

        // recompute flow from vectors
        compute_global_flow(predicted_flow_vectors, opticflow, result);
        free(predicted_flow_vectors);
      } else {
        // bottom cam: just subtract a scaled version of the roll and pitch difference from the global flow vector:
        diff_flow_x = phi_diff * OPTICFLOW_CAMERA.camera_intrinsics.focal_x; // phi_diff works better than (cam_state->rates.p)
        diff_flow_y = theta_diff * OPTICFLOW_CAMERA.camera_intrinsics.focal_y;
        result->flow_der_x = result->flow_x - diff_flow_x * opticflow->subpixel_factor *
                             opticflow->derotation_correction_factor_x;
        result->flow_der_y = result->flow_y - diff_flow_y * opticflow->subpixel_factor *
                             opticflow->derotation_correction_factor_y;
      }
    }
  }

  // Velocity calculation
  // Right now this formula is under assumption that the flow only exist in the center axis of the camera.
  // TODO: Calculate the velocity more sophisticated, taking into account the drone's angle and the slope of the ground plane.
  // TODO: This is actually only correct for the bottom camera:
  result->vel_cam.x = (float)result->flow_der_x * result->fps * agl_dist_value_filtered /
                      (opticflow->subpixel_factor * OPTICFLOW_CAMERA.camera_intrinsics.focal_x);
  result->vel_cam.y = (float)result->flow_der_y * result->fps * agl_dist_value_filtered /
                      (opticflow->subpixel_factor * OPTICFLOW_CAMERA.camera_intrinsics.focal_y);
  result->vel_cam.z = result->divergence * result->fps * agl_dist_value_filtered;

  //Apply a  median filter to the velocity if wanted
  if (opticflow->median_filter == true) {
    UpdateMedianFilterVect3Float(vel_filt, result->vel_cam);
  }

  // Determine quality of noise measurement for state filter
  //TODO develop a noise model based on groundtruth
  //result->noise_measurement = 1 - (float)opticflow->tracked_cnt / ((float)opticflow->max_track_corners * 1.25f);
  result->noise_measurement = 0.25;
  result->tracked_cnt = opticflow->tracked_cnt;
  result->corner_cnt = opticflow->corner_cnt;

  image_switch(&opticflow->img_gray, &opticflow->prev_img_gray);
  free(vectors);

  return true;
}

static void remove_bad_vectors(struct opticflow_t *opticflow, struct flow_t *vectors)
{
  if(opticflow->feature_management){
    if(opticflow->corner_cnt_prev){
      // remove bad vectors from the previously tracked points first, do two pass
      uint16_t good_vectors[opticflow->corner_cnt_prev];
      uint16_t good_counter = 0;
      for (int16_t i = 0; i < (int16_t)opticflow->corner_cnt_prev; i++) {
        if (vectors[i].error < LARGE_FLOW_ERROR){
          good_vectors[good_counter++] = i;
        }
      }
      // get total bad for previous set of points
      for (int16_t i = 0; i + 1 < (int16_t)opticflow->corner_cnt_prev; i++) {
        if (vectors[i].error >= LARGE_FLOW_ERROR){
          good_counter--;
          // move last element into this position effectively deleting point
          opticflow->tracked_cnt--;
          memcpy(&(vectors[i]), &(vectors[good_vectors[good_counter]]), sizeof(struct flow_t));
          memcpy(&(opticflow->fast9_ret_corners[i]), &(opticflow->fast9_ret_corners[good_vectors[good_counter]]), sizeof(struct point_t));
          opticflow->corner_cnt_prev--;
          memcpy(&(opticflow->fast9_ret_corners_prev[i]), &(opticflow->fast9_ret_corners_prev[good_vectors[good_counter]]), sizeof(struct point_t));
          // don't need to check this again, we know it's good, so no i--
        }
      }
    }

    // remove all rest of the bad vectors
    uint16_t start = opticflow->corner_cnt_prev >= 1 ? opticflow->corner_cnt_prev - 1 : 0;
    for (int16_t i = start; i + 1 < (int16_t)opticflow->tracked_cnt; i++) {
      if (vectors[i].error >= LARGE_FLOW_ERROR){
        // move last element into this position effectively deleting point
        opticflow->tracked_cnt--;
        memcpy(&(vectors[i]), &(vectors[opticflow->tracked_cnt]), sizeof(struct flow_t));
        memcpy(&(opticflow->fast9_ret_corners[i]), &(opticflow->fast9_ret_corners[opticflow->tracked_cnt]), sizeof(struct point_t));
        i--;
      }
    }
    // check final vector
    if (opticflow->corner_cnt_prev && vectors[opticflow->corner_cnt_prev - 1].error >= LARGE_FLOW_ERROR){
      opticflow->corner_cnt_prev--;
    }
  } else {
    // remove all bad vectors
    for (int16_t i = 0; i + 1 < (int16_t)opticflow->tracked_cnt; i++) {
      if (vectors[i].error >= LARGE_FLOW_ERROR){
        // move last element into this position effectively deleting point
        opticflow->tracked_cnt--;
        memcpy(&(vectors[i]), &(vectors[opticflow->tracked_cnt]), sizeof(struct flow_t));
        i--;
      }
    }
  }

  // check final vector
  if (opticflow->tracked_cnt && vectors[opticflow->tracked_cnt - 1].error >= LARGE_FLOW_ERROR){
    // move last element into this position effectively deleting point
    opticflow->tracked_cnt--;
  }
}
/*
 * Will check if there is a corner in the vicinity of that predicted by the flow
 */
static void check_tracked_points_corners(struct opticflow_t *opticflow, struct flow_t *vectors)
{
  uint16_t local_roi[4];  // local search roi
  uint16_t local_corners_size = 9;  // max corners to find in roi
  struct point_t *local_corners = calloc(local_corners_size, sizeof(struct point_t)); // storage for corners found

  for (int16_t i = 0; i < opticflow->tracked_cnt; i++) {
    if (vectors[i].error >= LARGE_FLOW_ERROR){
      continue;
    }

    // update corner location based on optical flow estimate of all successfully tracked points
    opticflow->fast9_ret_corners[i].x = (uint32_t)roundf((float)(vectors[i].pos.x + vectors[i].flow_x)/opticflow->subpixel_factor);
    opticflow->fast9_ret_corners[i].y = (uint32_t)roundf((float)(vectors[i].pos.y + vectors[i].flow_y)/opticflow->subpixel_factor);

    // check if point not too close to image edge
    static uint8_t window = 2;
    if (opticflow->fast9_ret_corners[i].x <= window || opticflow->fast9_ret_corners[i].y <= window ||
        opticflow->fast9_ret_corners[i].x + window >= opticflow->img_gray.w || opticflow->fast9_ret_corners[i].y + window >= opticflow->img_gray.h){
      vectors[i].error = LARGE_FLOW_ERROR;
      continue;
    }

    // define roi for corner search
    local_roi[0] = opticflow->fast9_ret_corners[i].x - window;
    local_roi[1] = opticflow->fast9_ret_corners[i].y - window;
    local_roi[2] = opticflow->fast9_ret_corners[i].x + window;
    local_roi[3] = opticflow->fast9_ret_corners[i].y + window;

    // try to redetect corners in search area
    uint16_t local_corners_found = 0;
    fast9_detect(&opticflow->img_gray, opticflow->fast9_threshold, 1, 0, 0,
        &local_corners_found, &local_corners_size, &local_corners, local_roi);

    if (local_corners_found) {
      // update corner location for next iteration
      float min_dist_squared = (2 * window + 1) * (2 * window + 1);
      float diff_x, diff_y, dist_squared;
      uint16_t min_ind = 0;
      for(uint16_t j = 0; j < local_corners_found; j++){
        diff_x = (float)opticflow->fast9_ret_corners[i].x - (float)local_corners[j].x;
        diff_y = (float)opticflow->fast9_ret_corners[i].y - (float)local_corners[j].y;
        dist_squared = diff_x*diff_x + diff_y*diff_y;
        if (dist_squared < min_dist_squared){
          min_dist_squared = dist_squared;
          min_ind = j;
        }
      }
      // Update fast corners
      opticflow->fast9_ret_corners[i].x = local_corners[min_ind].x;
      opticflow->fast9_ret_corners[i].y = local_corners[min_ind].y;

      // update vectors with local_corners[min_ind].x - opticflow->fast9_ret_corners[i].x;
      //vectors[i].flow_x += local_corners[min_ind].x*opticflow->subpixel_factor - opticflow->fast9_ret_corners[i].x_full;
      //vectors[i].flow_y += local_corners[min_ind].y*opticflow->subpixel_factor - opticflow->fast9_ret_corners[i].y_full;
    } else {
      vectors[i].error = LARGE_FLOW_ERROR;
    }
  }
  free(local_corners);
}

void check_back_flow(struct opticflow_t *opticflow, struct flow_t *vectors)
{
  // copy corners
  struct point_t *my_corners = calloc(opticflow->tracked_cnt, sizeof(struct point_t));

  // initialize corners at the tracked positions:
  for (int i = 0; i < opticflow->tracked_cnt; i++) {
    my_corners[i].x = (uint32_t)roundf((float)(vectors[i].pos.x + vectors[i].flow_x)/opticflow->subpixel_factor);
    my_corners[i].y = (uint32_t)roundf((float)(vectors[i].pos.y + vectors[i].flow_y)/opticflow->subpixel_factor);
  }

  // present the images in the opposite order:
  uint16_t back_track_cnt = opticflow->tracked_cnt;
  struct flow_t *back_vectors = opticFlowLK(&opticflow->prev_img_gray, &opticflow->img_gray, my_corners,
                                &back_track_cnt, opticflow->window_size / 2, opticflow->subpixel_factor, opticflow->max_iterations,
                                opticflow->threshold_vec, opticflow->tracked_cnt, opticflow->pyramid_level, 1);

  float back_x, back_y, diff_x, diff_y, dist_squared;
  float back_track_threshold = 200;

  for (int i = 0; i < opticflow->tracked_cnt; i++) {
    if (vectors[i].error >= LARGE_FLOW_ERROR){
      continue;
    }
    if (back_vectors[i].error < LARGE_FLOW_ERROR) {
      back_x = back_vectors[i].pos.x + back_vectors[i].flow_x;
      back_y = back_vectors[i].pos.y + back_vectors[i].flow_y;
      diff_x = back_x - vectors[i].pos.x;
      diff_y = back_y - vectors[i].pos.y;
      dist_squared = diff_x * diff_x + diff_y * diff_y;
      if (dist_squared > back_track_threshold) {
        vectors[i].error = LARGE_FLOW_ERROR;
      }
    } else {
      vectors[i].error = LARGE_FLOW_ERROR;
    }
  }
  free(my_corners);
  free(back_vectors);
}

/*
 * Predict flow vectors by means of the rotation rates:
 */
static struct flow_t *predict_flow_vectors(struct flow_t *flow_vectors, uint16_t n_points, float phi_diff,
    float theta_diff, float psi_diff, struct opticflow_t *opticflow)
{

  // reserve memory for the predicted flow vectors:
  struct flow_t *predicted_flow_vectors = calloc(n_points, sizeof(struct flow_t));

  float K[9] = {OPTICFLOW_CAMERA.camera_intrinsics.focal_x, 0.0f, OPTICFLOW_CAMERA.camera_intrinsics.center_x,
                0.0f, OPTICFLOW_CAMERA.camera_intrinsics.focal_y, OPTICFLOW_CAMERA.camera_intrinsics.center_y,
                0.0f, 0.0f, 1.0f
               };
  // TODO: make an option to not do distortion / undistortion (Dhane_k = 1)
  float k = OPTICFLOW_CAMERA.camera_intrinsics.Dhane_k;

  float A, B, C; // as in Longuet-Higgins

  if (strcmp(OPTICFLOW_CAMERA.dev_name, "/dev/video1") == 0) {
    // specific for the x,y swapped Bebop 2 images:
    A = -psi_diff;
    B = theta_diff;
    C = phi_diff;
  } else {
    A = theta_diff;
    B = phi_diff;
    C = psi_diff;
  }

  float x_n, y_n;
  float x_n_new, y_n_new, x_pix_new, y_pix_new;
  float predicted_flow_x, predicted_flow_y;
  for (uint16_t i = 0; i < n_points; i++) {
    // the from-coordinate is always the same:
    predicted_flow_vectors[i].pos.x = flow_vectors[i].pos.x;
    predicted_flow_vectors[i].pos.y = flow_vectors[i].pos.y;

    bool success = distorted_pixels_to_normalized_coords((float)flow_vectors[i].pos.x / opticflow->subpixel_factor,
                   (float)flow_vectors[i].pos.y / opticflow->subpixel_factor, &x_n, &y_n, k, K);
    if (success) {
      // predict flow as in a linear pinhole camera model:
      predicted_flow_x = A * x_n * y_n - B * x_n * x_n - B + C * y_n;
      predicted_flow_y = -C * x_n + A + A * y_n * y_n - B * x_n * y_n;

      x_n_new = x_n + predicted_flow_x;
      y_n_new = y_n + predicted_flow_y;

      success = normalized_coords_to_distorted_pixels(x_n_new, y_n_new, &x_pix_new, &y_pix_new, k, K);

      if (success) {
        predicted_flow_vectors[i].flow_x = (int16_t)(x_pix_new * opticflow->subpixel_factor - (float)flow_vectors[i].pos.x);
        predicted_flow_vectors[i].flow_y = (int16_t)(y_pix_new * opticflow->subpixel_factor - (float)flow_vectors[i].pos.y);
        predicted_flow_vectors[i].error = 0;
      } else {
        predicted_flow_vectors[i].flow_x = 0;
        predicted_flow_vectors[i].flow_y = 0;
        predicted_flow_vectors[i].error = LARGE_FLOW_ERROR;
      }
    } else {
      predicted_flow_vectors[i].flow_x = 0;
      predicted_flow_vectors[i].flow_y = 0;
      predicted_flow_vectors[i].error = LARGE_FLOW_ERROR;
    }
  }
  return predicted_flow_vectors;
}


/* manage_flow_features - Update list of corners to be tracked by LK
 * Remembers previous points and tries to find new points in less dense
 * areas of the image first.
 */
static void manage_flow_features(struct opticflow_t *opticflow, uint16_t *roi)
{
  opticflow->corner_cnt = opticflow->tracked_cnt;
  // remove tracked corners that have drifted too close together or outside of roi
  for (int16_t c1 = 0; c1 + 1 < (int16_t)opticflow->corner_cnt; c1++) {
    if (opticflow->fast9_ret_corners[c1].x < roi[0] || opticflow->fast9_ret_corners[c1].x > roi[2]||
        opticflow->fast9_ret_corners[c1].y < roi[1] || opticflow->fast9_ret_corners[c1].y > roi[3]){
      // corner has drifted out of roi
      opticflow->corner_cnt--;
      memcpy(&(opticflow->fast9_ret_corners[c1]), &(opticflow->fast9_ret_corners[opticflow->corner_cnt]), sizeof(struct point_t));
      // check this location again
      c1--;
      continue;
    }
    for (int16_t i = c1 + 1; i < opticflow->corner_cnt; i++) {
      if (abs(opticflow->fast9_ret_corners[c1].x - opticflow->fast9_ret_corners[i].x) < opticflow->fast9_min_distance / 2
          && abs(opticflow->fast9_ret_corners[c1].y - opticflow->fast9_ret_corners[i].y) < opticflow->fast9_min_distance / 2) {
        // if too close, replace the corner with the last one in the list, effectively removing the point:
        opticflow->corner_cnt--;
        memcpy(&(opticflow->fast9_ret_corners[c1]), &(opticflow->fast9_ret_corners[opticflow->corner_cnt]), sizeof(struct point_t));
        // check this location again
        c1--;
        // no further checking required for the removed corner
        break;
      }
    }
  }
  if (opticflow->corner_cnt &&
      (opticflow->fast9_ret_corners[opticflow->corner_cnt -1].x < roi[0] || opticflow->fast9_ret_corners[opticflow->corner_cnt - 1].x > roi[2]||
        opticflow->fast9_ret_corners[opticflow->corner_cnt -1].y < roi[1] || opticflow->fast9_ret_corners[opticflow->corner_cnt - 1].y > roi[3])){
    opticflow->corner_cnt--;
  }

  // make copy of corners that will be tracked from previous step
  if (opticflow->fast9_rsize > opticflow->fast9_rsize_prev){
    opticflow->fast9_ret_corners_prev = realloc(opticflow->fast9_ret_corners_prev, opticflow->fast9_rsize * sizeof(struct point_t));
  }
  memcpy(opticflow->fast9_ret_corners_prev, opticflow->fast9_ret_corners, opticflow->corner_cnt * sizeof(struct point_t));
  opticflow->corner_cnt_prev = opticflow->corner_cnt;

  if (opticflow->corner_cnt > opticflow->max_track_corners / 2){
    return;
  }
  // allocation new corners
  // no need for "per region" re-detection when there are no previous corners
  if ((!opticflow->fast9_region_detect) || (opticflow->corner_cnt == 0)) {
    if (opticflow->corner_method == EXHAUSTIVE_FAST) {
      fast9_detect(&opticflow->prev_img_gray, opticflow->fast9_threshold, opticflow->fast9_min_distance,
                   opticflow->fast9_padding, opticflow->fast9_padding, &opticflow->corner_cnt,
                   &opticflow->fast9_rsize, &opticflow->fast9_ret_corners, roi);
    } else if (opticflow->corner_method == ACT_FAST) {
        act_fast(&opticflow->prev_img_gray, opticflow->fast9_threshold, &opticflow->corner_cnt,
                 &opticflow->fast9_ret_corners, opticflow->max_track_corners, n_time_steps,
                 opticflow->actfast_long_step, opticflow->actfast_short_step, opticflow->actfast_min_gradient,
                 opticflow->actfast_gradient_method);
    }
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
    for (uint16_t i = 0; i < opticflow->corner_cnt; i++) {
      region_index = (opticflow->fast9_ret_corners[i].x * root_regions / opticflow->prev_img_gray.w
         + root_regions * (opticflow->fast9_ret_corners[i].y * root_regions / opticflow->prev_img_gray.h));
      region_index = (region_index < opticflow->fast9_num_regions) ? region_index : opticflow->fast9_num_regions - 1;
      region_count[region_index][0]++;
    }

    //sorting region_count array according to first column (number of corners).
    qsort(region_count, opticflow->fast9_num_regions, sizeof(region_count[0]), cmp_array);

    // Detecting corners from the region with the less to the one with the most, until a desired total is reached.
    for (uint16_t i = 0; i < opticflow->fast9_num_regions && opticflow->corner_cnt < 2 * opticflow->max_track_corners; i++) {
      // Find the boundaries of the region of interest
      roi[0] = (region_count[i][1] % root_regions) * (opticflow->prev_img_gray.w / root_regions);
      roi[1] = (region_count[i][1] / root_regions) * (opticflow->prev_img_gray.h / root_regions);
      roi[2] = roi[0] + (opticflow->prev_img_gray.w / root_regions);
      roi[3] = roi[1] + (opticflow->prev_img_gray.h / root_regions);

      struct point_t *new_corners = calloc(opticflow->fast9_rsize, sizeof(struct point_t));
      uint16_t new_count = 0;

      fast9_detect(&opticflow->prev_img_gray, opticflow->fast9_threshold, opticflow->fast9_min_distance,
                   opticflow->fast9_padding, opticflow->fast9_padding, &new_count,
                   &opticflow->fast9_rsize, &new_corners, roi);

      // check that no identified points already exist in list
      for (uint16_t j = 0; j < new_count; j++) {
        bool exists = false;
        for (uint16_t k = 0; k < opticflow->corner_cnt; k++) {
          if (abs((int16_t)new_corners[j].x - (int16_t)opticflow->fast9_ret_corners[k].x) < (int16_t)opticflow->fast9_min_distance
              && abs((int16_t)new_corners[j].y - (int16_t)opticflow->fast9_ret_corners[k].y) < (int16_t)
              opticflow->fast9_min_distance) {
            exists = true;
            break;
          }
        }
        if (!exists) {
          memcpy(&(opticflow->fast9_ret_corners[opticflow->corner_cnt]), &(new_corners[j]), sizeof(struct point_t));
          opticflow->corner_cnt++;
          if (opticflow->corner_cnt >= opticflow->fast9_rsize) {
            break;
          }
        }
      }

      free(new_corners);
    }
    for (uint16_t i = 0; i < opticflow->fast9_num_regions; i++) {
      free(region_count[i]);
    }
    free(region_count);
  }
}

void compute_global_flow(struct flow_t *vectors, struct opticflow_t *opticflow, struct opticflow_result_t *result)
{
  if (opticflow->tracked_cnt < 1){
    result->flow_x = 0;
    result->flow_y = 0;
    return;
  }

  // compute flow from vectors
  if(0){//opticflow->tracked_cnt > opticflow->max_track_corners / 2){
    // if you have enough points, a median is a good estimate of flow
    qsort(vectors, result->tracked_cnt, sizeof(struct flow_t), cmp_flow);
    if (result->tracked_cnt % 2) {
      // Take the median point
      result->flow_x = vectors[result->tracked_cnt / 2].flow_x;
      result->flow_y = vectors[result->tracked_cnt / 2].flow_y;
    } else {
      // Take the average of the 2 median points
      result->flow_x = (vectors[result->tracked_cnt / 2 - 1].flow_x + vectors[result->tracked_cnt / 2].flow_x) / 2;
      result->flow_y = (vectors[result->tracked_cnt / 2 - 1].flow_y + vectors[result->tracked_cnt / 2].flow_y) / 2;
    }
  } else {
    // determine the mean for the vector:
    float sum_x = 0.f, sum_y = 0.f;
    for (uint32_t i = 0; i < opticflow->tracked_cnt; i++) {
      sum_x += (float)vectors[i].flow_x;
      sum_y += (float)vectors[i].flow_y;
    }
    result->flow_x = (int32_t)(sum_x / opticflow->tracked_cnt);
    result->flow_y = (int32_t)(sum_y / opticflow->tracked_cnt);
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
                            OPTICFLOW_CAMERA.camera_intrinsics.focal_x * opticflow->derotation_correction_factor_x);
    der_shift_y = (int16_t)((edge_hist[current_frame_nr].eulers.theta - edge_hist[previous_frame_nr[1]].eulers.theta) *
                            OPTICFLOW_CAMERA.camera_intrinsics.focal_y * opticflow->derotation_correction_factor_y);
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
  opticflow->corner_cnt = getAmountPeaks(edge_hist_x, 500 , img->w);
  opticflow->tracked_cnt = getAmountPeaks(edge_hist_x, 500 , img->w);
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
  result->vel_cam.x = edgeflow.flow_x * fps_x * agl_dist_value_filtered * OPTICFLOW_CAMERA.camera_intrinsics.focal_x /
                      RES;
  result->vel_cam.y = edgeflow.flow_y * fps_y * agl_dist_value_filtered * OPTICFLOW_CAMERA.camera_intrinsics.focal_y /
                      RES;
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
  const uint16_t *pa = (const uint16_t *)a;
  const uint16_t *pb = (const uint16_t *)b;
  return pa[0] - pb[0];
}

// Initialize object to be tracked
void init_object_tracking(struct opticflow_t *opticflow, struct object_tracker_t *tracker){
  tracker->roi_defined = false;

  tracker->roi_centriod_x = 120;
  tracker->roi_centriod_y = 120;
  tracker->roi_h = 160;
  tracker->roi_w = 160;

  tracker->roi[0] = 1;
  tracker->roi[1] = 1;
  tracker->roi[2] = OPTICFLOW_CAMERA.output_size.w-2;
  tracker->roi[3] = OPTICFLOW_CAMERA.output_size.h-2;

  tracker->control_active = false;
}

#include "math.h"
#include "guidance/guidance_h.h"
#include "guidance/guidance_v.h"
#include "subsystems/datalink/telemetry.h"
static void update_object_roi(struct opticflow_t *opticflow, struct image_t *img, struct opticflow_result_t *result, struct object_tracker_t *tracker)
{
  if(tracker->roi_defined){
    // if not enough points to track use regular flow instead of corners
    if(opticflow->corner_cnt_prev > 3){
        // let us separate change in point locations into change due to motion and change due to lost/added points
      float flow_sum_x = 0.f, flow_sum_y = 0.f;  // sum of flow
      float centroid_sum_x = 0.f, centroid_sum_y = 0.f;
      float centroid_sum_x_prev = 0.f, centroid_sum_y_prev = 0.f;
      for (uint16_t i = 0; i < opticflow->corner_cnt_prev; i++) {
        // get displacement of point
        flow_sum_x += (int32_t)opticflow->fast9_ret_corners[i].x - (int32_t)opticflow->fast9_ret_corners_prev[i].x;
        flow_sum_y += (int32_t)opticflow->fast9_ret_corners[i].y - (int32_t)opticflow->fast9_ret_corners_prev[i].y;

        // get sum of distance
        centroid_sum_x += opticflow->fast9_ret_corners[i].x;
        centroid_sum_y += opticflow->fast9_ret_corners[i].y;

        centroid_sum_x_prev += opticflow->fast9_ret_corners_prev[i].x;
        centroid_sum_y_prev += opticflow->fast9_ret_corners_prev[i].y;
      }
      // update roi location
      tracker->roi_centriod_x += flow_sum_x / opticflow->corner_cnt_prev;
      tracker->roi_centriod_y += flow_sum_y / opticflow->corner_cnt_prev;

      float centroid_x = centroid_sum_x / opticflow->corner_cnt_prev;
      float centroid_y = centroid_sum_y / opticflow->corner_cnt_prev;

      float centroid_x_prev = centroid_sum_x_prev / opticflow->corner_cnt_prev;
      float centroid_y_prev = centroid_sum_y_prev / opticflow->corner_cnt_prev;

      // compute change in 1st moment of inertia
      float change_sum = 0.f;
      float diff_x, diff_y;
      float dist_1, dist_2;
      uint32_t sum = 0;
      for (uint16_t i = 0; i < opticflow->corner_cnt_prev; i++) {
        diff_x = opticflow->fast9_ret_corners_prev[i].x - centroid_x_prev;
        diff_y = opticflow->fast9_ret_corners_prev[i].y - centroid_y_prev;
        dist_1 = sqrtf(diff_x * diff_x + diff_y * diff_y);

        diff_x = opticflow->fast9_ret_corners[i].x - centroid_x;
        diff_y = opticflow->fast9_ret_corners[i].y - centroid_y;
        dist_2 = sqrtf(diff_x * diff_x + diff_y * diff_y);
        if (dist_1 > 1e-5){
          change_sum +=  dist_2 / dist_1;
          sum++;
        }
      }

      if (sum){
        // update roi size
        float scaler = change_sum/sum;
        //printf("x %.2f, y %.2f, w %.2f, h %.2f, change %f %f, %f %f\n", tracker->roi_centriod_x, tracker->roi_centriod_y, tracker->roi_w, tracker->roi_h,
        //    change_sum/sum, 1.f + result->div_size, tracker->roi_w*scaler, tracker->roi_h*scaler);
        // outlier detection
        if (fabsf(scaler - (1.f + result->div_size)) / (1.f + result->div_size) < 0.1f){
          //tracker->roi_w *= scaler;
          //tracker->roi_h *= scaler;
        }
      }
    } else {
      // use regular flow to update image relative location
      tracker->roi_centriod_x += result->flow_x;
      tracker->roi_centriod_y += result->flow_y;

      //tracker->roi_w *= 1.f + result->div_size;
      //tracker->roi_h *= 1.f + result->div_size;
    }

    // update image roi
    tracker->roi[0] = (uint16_t)Min(img->w-1, Max(0.f, tracker->roi_centriod_x - tracker->roi_w/2));
    tracker->roi[1] = (uint16_t)Min(img->h-1, Max(0.f, tracker->roi_centriod_y - tracker->roi_h/2));
    tracker->roi[2] = (uint16_t)Max(0.f, Min(img->w-1, tracker->roi_centriod_x + tracker->roi_w/2));
    tracker->roi[3] = (uint16_t)Max(0.f, Min(img->h-1, tracker->roi_centriod_y + tracker->roi_h/2));
    if(tracker->roi[2] - tracker->roi[0] < img->w / 6 || tracker->roi[3] - tracker->roi[1] < img->h / 6){
      // roi too small, use entire frame
      tracker->roi[0] = 1;
      tracker->roi[1] = 1;
      tracker->roi[2] = img->w-2;
      tracker->roi[3] = img->h-2;
    }

    // draw roi on image
    struct point_t roi_to_show[4];
    roi_to_show[0].x = tracker->roi[0];
    roi_to_show[0].y = tracker->roi[1];
    roi_to_show[1].x = tracker->roi[0];
    roi_to_show[1].y = tracker->roi[3];
    roi_to_show[2].x = tracker->roi[2];
    roi_to_show[2].y = tracker->roi[1];
    roi_to_show[3].x = tracker->roi[2];
    roi_to_show[3].y = tracker->roi[3];

#if OPTICFLOW_SHOW_CORNERS
    static uint8_t white[4] = {127, 255, 127, 255};
    image_show_points_color(img, roi_to_show, 4, white);
#endif

    // get position relative to frame center
    float x = tracker->roi_centriod_x - OPTICFLOW_CAMERA.camera_intrinsics.center_x;
    float y = tracker->roi_centriod_y - OPTICFLOW_CAMERA.camera_intrinsics.center_y;

    // TODO implement body to cam rotation

    //int32_t f = (int32_t)roundf(sqrtf(OPTICFLOW_CAMERA.camera_intrinsics.focal_x * OPTICFLOW_CAMERA.camera_intrinsics.focal_x +
        //OPTICFLOW_CAMERA.camera_intrinsics.focal_y * OPTICFLOW_CAMERA.camera_intrinsics.focal_y));
    int32_t f = 300;
    struct FloatVect3 img_coord = {-y, x, f};
    struct FloatVect3 virt_coord;

    // Body <-> LTP, no psi
    struct FloatRMat ltp2body;
    FLOAT_MAT33_ZERO(ltp2body);
    struct FloatEulers *eulers = stateGetNedToBodyEulers_f();
    const float sphi   = sinf(eulers->phi);
    const float cphi   = cosf(eulers->phi);
    const float stheta = sinf(eulers->theta);
    const float ctheta = cosf(eulers->theta);

    RMAT_ELMT(ltp2body, 0, 0) = ctheta;
    RMAT_ELMT(ltp2body, 0, 2) = -stheta;
    RMAT_ELMT(ltp2body, 1, 0) = sphi * stheta;
    RMAT_ELMT(ltp2body, 1, 1) = cphi;
    RMAT_ELMT(ltp2body, 1, 2) = sphi * ctheta;
    RMAT_ELMT(ltp2body, 2, 0) = cphi * stheta;
    RMAT_ELMT(ltp2body, 2, 1) = -sphi;
    RMAT_ELMT(ltp2body, 2, 2) = cphi * ctheta;

    float_rmat_transp_vmult(&virt_coord, &ltp2body, &img_coord);

    struct FloatVect2 control_error = {virt_coord.x / img->w, virt_coord.y / img->h};

    static const float gain = 0.5f;
    static const float deadband = 0.05f;
    static int center_count = 0;
    if(tracker->control_active){
      if (fabsf(control_error.x) < deadband) {
        control_error.x = 0.f;
      }
      if (fabsf(control_error.y) < deadband) {
        control_error.y = 0.f;
      }
      guidance_h_set_guided_body_vel(control_error.x * gain * stateGetPositionEnu_f()->z, control_error.y * gain * stateGetPositionEnu_f()->z);
      if (control_error.x < deadband && control_error.y < deadband){
        if (center_count++ > 5){
          //guidance_v_set_guided_vz(0.25f);
        }
      } else {
        if (center_count){
          center_count--;
        } else {
          //guidance_v_set_guided_vz(0.f);
        }
      }
    }
    DOWNLINK_SEND_OBJECT_TRACKING(DefaultChannel, DefaultDevice,
                                 &x, &y,
                                 &tracker->roi_w, &tracker->roi_h,
                                 &virt_coord.x, &virt_coord.y);
  }

  /*printf("(%f %f), (%f %f), (%f %f)\n", img_coord.y, img_coord.x, virt_coord.y, virt_coord.x,
      stateGetNedToBodyEulers_f()->phi,stateGetNedToBodyEulers_f()->theta);*/

}
