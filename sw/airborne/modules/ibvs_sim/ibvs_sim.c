/*
 * Copyright (C) Jari Blom
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/ibvs_sim/ibvs_sim.c"
 * @author Jari Blom
 * Simulate Image Based Visual Servoing
 */

/**
 * Camera reference frame as well as virtual camera reference frame
 * is defined from 0 to 1
 * World Frame is ENU
 */

#include "std.h"
#include <stdio.h>

#include "modules/ibvs_sim/ibvs_sim.h"
#include "firmwares/rotorcraft/guidance.h"
#include "state.h"
#include <math.h>
#include "cv.h"
// for measuring time
#include "mcu_periph/sys_time.h"
// For vertical landing
#include "firmwares/rotorcraft/stabilization.h"
#include "subsystems/datalink/telemetry.h"
#include "modules/ctrl/optical_flow_landing.h"
#include "subsystems/abi.h"
#include "paparazzi.h"
#include "autopilot.h"
#include "subsystems/navigation/common_flight_plan.h"
#include "generated/airframe.h"

#ifndef IBVS_FPS
#define IBVS_FPS 0       ///< Default FPS (zero means run at camera fps)
#endif

#ifndef IBVS_FX
#define IBVS_FX 343.1211       ///< Default FPS (zero means run at camera fps)
#endif

#ifndef IBVS_FY
#define IBVS_FY 348.5053       ///< Default FPS (zero means run at camera fps)
#endif

#ifndef IBVS_X_GAIN
#define IBVS_X_GAIN 0.005		  // Control gain for servoing in the x-direction
#endif

#ifndef IBVS_Y_GAIN
#define IBVS_Y_GAIN -0.005       // Control gain for servoing in the y-direction
#endif

#ifndef IBVS_Z_GAIN
#define IBVS_Z_GAIN 0.1        // Control gain for servoing in the z-direction
#endif

#ifndef IBVS_YAW_GAIN
#define IBVS_YAW_GAIN 0.001        // Control gain for servoing around the z-axis
#endif

#ifndef IBVS_ON
#define IBVS_ON FALSE        // Setting for taking over flight plan by ibvs algorithm
#endif

#ifndef IBVS_SET_GUIDANCE
#define IBVS_SET_GUIDANCE FALSE        // Setting for taking over flight plan by ibvs algorithm
#endif

#ifndef NDT_VZ
#define NDT_VZ 10        // Setting for taking over flight plan by ibvs algorithm
#endif

// From optical_flow_landing.c
// for exponential gain landing, gain increase per second during the first (hover) phase:
#define INCREASE_GAIN_PER_SECOND 0.0000009
// From paparazzi.h
#define MAX_PPRZ 9600

#ifndef OFL_PGAIN
#define OFL_PGAIN 0.000008
#endif

#ifndef OFL_IGAIN
#define OFL_IGAIN 0.0
#endif

#ifndef OFL_DGAIN
#define OFL_DGAIN 0.0
#endif
/*

// For exponential
#ifndef OFL_COV_SETPOINT
#define OFL_COV_SETPOINT -0.0015
#endif
*/

// For adaptive
#ifndef OFL_COV_SETPOINT
#define OFL_COV_SETPOINT -0.0005
#endif

#ifndef OFL_COV_LANDING_LIMIT
#define OFL_COV_LANDING_LIMIT 2.2
#endif

#ifndef OFL_LP_CONST
#define OFL_LP_CONST 0.02
#endif

#ifndef OFL_COV_METHOD
#define OFL_COV_METHOD 0
#endif

// number of time steps used for calculating the covariance (oscillations)
#ifndef OFL_COV_WINDOW_SIZE
#define OFL_COV_WINDOW_SIZE 30
#endif

#ifndef OFL_P_LAND_THRESHOLD
#define OFL_P_LAND_THRESHOLD 0.0000007
#endif

/*
#ifndef OFL_P_LAND_THRESHOLD
#define OFL_P_LAND_THRESHOLD 0.0
#endif
*/

#ifndef OFL_ELC_OSCILLATE
#define OFL_ELC_OSCILLATE true
#endif

/* Default sonar/agl to use */
#ifndef OFL_AGL_ID
#define OFL_AGL_ID ABI_BROADCAST
#endif
PRINT_CONFIG_VAR(OFL_AGL_ID)

/* Use optical flow estimates */
#ifndef OFL_OPTICAL_FLOW_ID
#define OFL_OPTICAL_FLOW_ID ABI_BROADCAST
#endif
PRINT_CONFIG_VAR(OFL_OPTICAL_FLOW_ID)

// Constants
// minimum value of the P-gain for divergence control
// adaptive control / exponential gain control will not be able to go lower
#define MINIMUM_GAIN 0.0000005
//#define MINIMUM_GAIN 0.0

// Define variables and functions only used in this file
// Goal values for features
uint16_t x_gstar = 0;
uint16_t y_gstar = 0;
float s3_star = 1;
float alpha_star = 0.1; // Orientation in mrad
uint16_t mu_2002star = 500; //
float v_xstar, v_ystar, v_zstar, r_star;
float dwdtw_star = 1.0; // Desired size increase per second for the ROI
float v_zadd = 0.;
uint32_t iter = 0;

// Camera parameters
const float l1 = IBVS_FX;
const float l2 = IBVS_FY;
uint32_t x_center = 120;
uint32_t y_center = 120;
int i,j,k,l;
struct FloatEulers *euler_angles;
float res1[3][1];
float res2[2][1];
float LX[3][1];
float pv[2][1];
struct Tracked_object object_to_track;
// struct containing most relevant parameters
struct OpticalFlowLanding of_landing_ctrl;
float item;
float x_gain = -1*IBVS_X_GAIN;
float y_gain = -1*IBVS_Y_GAIN;
float z_gain = -1*IBVS_Z_GAIN;
float yaw_gain = IBVS_YAW_GAIN;
float div_factor = -1.28f; // magic number comprising field of view etc.


// variables retained between module calls for landing
float divergence_vision;
float divergence_vision_dt;
float normalized_thrust;
float cov_div;
float pstate, pused;
float istate;
float dstate;
float vision_time,  prev_vision_time;
bool landing;
float previous_cov_err;
float thrust_set;
float divergence_setpoint;
// for the exponentially decreasing gain strategy:
int32_t elc_phase;
uint32_t elc_time_start;
float elc_p_gain_start, elc_i_gain_start,  elc_d_gain_start, elc_p_gain_off_min;
int32_t count_covdiv;
float lp_cov_div;
bool bool_cov_array_filled;
bool setting_average_thrust;
float p_x_ratio; // ratio between vertical and horizontal gain
bool to_phase_2; // Indicating we're intializing the p_x_ratio

static abi_event agl_ev; ///< The altitude ABI event
static abi_event optical_flow_ev;
float thrust_history[OFL_COV_WINDOW_SIZE];
float divergence_history[OFL_COV_WINDOW_SIZE];
float past_divergence_history[OFL_COV_WINDOW_SIZE];
uint32_t ind_hist;
uint8_t cov_array_filled;
float time_at_filled;
float p_x_ratio;
bool set_x_gain; // Are we changing x_gain based on z_gain

static void multiply3x33x1(float mat1[3][3],float mat2[3][1],float res[3][1]);
static void multiply2x33x1(float mat1[2][3],float mat2[3][1],float res[2][1]);
static struct image_t *calc_ibvs_control(struct image_t *img);
static void calc_frame_ibvs_control(struct opticflow_t *opticflow,struct Tracked_object *object_to_track);
// Functions copied from optical_flow_landing.c
static void vertical_ctrl_module_init(void);
static float final_landing_procedure(struct opticflow_t *opticflow);
static float PID_divergence_control(float divergence_setpoint, float P, float I, float D, float dt,
    struct opticflow_t *opticflow, struct OpticalFlowLanding of_landing_ctrl);
static void reset_all_vars(void);
static void set_cov_div(float,struct opticflow_t *opticflow);
static void update_errors(float err, float dt);
// Callback function of the ground altitude
static void vertical_ctrl_agl_cb(uint8_t sender_id, float distance);
// Callback function of the optical flow estimate:
static void vertical_ctrl_optical_flow_cb(uint8_t sender_id, uint32_t stamp, int16_t flow_x,
                                   int16_t flow_y, int16_t flow_der_x, int16_t flow_der_y, float quality, float size_divergence);



void ibvs_sim_init()
{
	// Initialize the opticflow calculation
	// There should also be a check if we got a result and if that result is usefull
    // ibvs_got_result = false;

  cv_add_to_device(&IBVS_CAMERA, calc_ibvs_control, IBVS_FPS);

    /*we should probably have something similar to this for IBVS
	#if PERIODIC_TELEMETRY
	  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_OPTIC_FLOW_EST, opticflow_telem_send);
	#endif
    */

}

void ibvs_sim_periodic()
{

}

// This function multiplies mat1 and mat2
// Only works for 3x3 * 3x3, because of the function definition
void multiply3x33x1(float mat1[3][3],float mat2[3][1], float res[3][1])
{
	for(int j1=0;j1<3;j1++){
		for(int i1=0;i1<1;i1++){
			item = 0.0;
			for(int k1=0;k1<3;k1++){
				item += mat1[j1][k1]*mat2[k1][i1];
			}
			res[j1][i1] = item;
		}
	}
}

// This function multiplies mat1 and mat2
// Only works for 2x3 * 3x3, because of the function definition
void multiply2x33x1(float mat1[2][3],float mat2[3][1],float res[2][1])
{
	for(int j1=0;j1<2;j1++){
		for(int i1=0;i1<1;i1++){
			item = 0.0;
			for(int k1=0;k1<3;k1++){
				item += mat1[j1][k1]*mat2[k1][i1];
			}
			res[j1][i1] = item;
		}
	}
}

struct image_t *calc_ibvs_control(struct image_t *img){

	calc_frame_ibvs_control(&opticflow,&object_to_track);

	return img;
}

// Actual IBVS
void calc_frame_ibvs_control(struct opticflow_t *opticflow,struct Tracked_object *object_to_track)
{
	if(opticflow->object_tracking){
		// Init object_tracking
		if(opticflow->ibvs_init){
		  if(opticflow->object_tracking_reset){
		    opticflow->object_tracking_reset = false;
		  }
		  else{
		    // Initialize L_matrix
        for(i=0;i<2;i++){
          for(j=0;j<3;j++){
            object_to_track->L_matrix[i][j] = 0;
          }
        }
        // Initialize object_to_track
        object_to_track->L_matrix[0][0] = l1;
        object_to_track->L_matrix[1][1] = l2;
        object_to_track->ibvs_go = IBVS_ON;
        object_to_track->set_guidance = IBVS_SET_GUIDANCE;
        object_to_track->vision_time = get_sys_time_float();
        object_to_track->prev_vision_time = object_to_track->vision_time;
        object_to_track->ibvs_go2 = false;
        object_to_track->ndt_vz = NDT_VZ;
        // Initialize vertical control
        vertical_ctrl_module_init();
        // Some settings
        opticflow->in_flight = true;
        opticflow->landing = false;
        opticflow->cov_div = 0.;
        time_at_filled = get_sys_time_float()*10;
        bool_cov_array_filled = false;
        setting_average_thrust = false;
        // Initialize result_to_write
        uint32_t max_size = 1000;
        to_phase_2 = false;
        set_x_gain = false;
        /*result_to_write->enu = calloc(max_size, sizeof(struct EnuCoor_f));
        result_to_write->venu = calloc(max_size, sizeof(struct EnuCoor_f));
        result_to_write->timestamp = calloc(max_size, sizeof(float));
        result_to_write->t_new_window = calloc(30, sizeof(float));*/
		  }
		  object_to_track->roiw_prev = opticflow->roiw;
      object_to_track->roih_prev = opticflow->roih;
		  opticflow->ibvs_init = false;
		}
		object_to_track->vision_time = get_sys_time_float();
		// Get dt
		// Might give errors for first time stamp directly after init
    float dto = object_to_track->vision_time - object_to_track->prev_vision_time;
    float lp_factor = dto / of_landing_ctrl.lp_const;
    // Check if new measurement received
    if (dto <= 1e-5f) {
      return;
    }

    /***********
    * Horizontal CONTROL
    ***********/
		// Define roi_cg in the camera frame, with (0,0) at the center of the frame and y positive upwards
		object_to_track->object_cg.x = opticflow->roi_center.x-x_center;
    object_to_track->object_cg.y = -1*(opticflow->roi_center.y-y_center);

		// Rotate to virtual camera frame
		euler_angles = stateGetNedToBodyEulers_f();
		float R[3][3] = {
				{cos( euler_angles->phi),0,sin( euler_angles->phi)},
				{sin( euler_angles->theta)*sin( euler_angles->phi), cos( euler_angles->theta),-sin( euler_angles->theta)*cos( euler_angles->phi)},
				{-sin( euler_angles->phi)*cos( euler_angles->theta), sin( euler_angles->theta), cos( euler_angles->theta)*cos( euler_angles->phi)}
		};


    float beta = l1*l2*cos( euler_angles->phi)*cos( euler_angles->theta)-
        object_to_track->object_cg.x*l2*sin( euler_angles->theta) +
        object_to_track->object_cg.y*l1*sin( euler_angles->phi)*cos( euler_angles->theta);
    // Fill LX
    LX[0][0] = l2*(int)object_to_track->object_cg.x;
    LX[1][0] = l1*(int)object_to_track->object_cg.y;
    LX[2][0] = l1*l2;

    multiply3x33x1(R,LX,res1);
    multiply2x33x1(object_to_track->L_matrix,res1,res2);
    pv[0][0] = res2[0][0]/beta;
    pv[1][0] = res2[1][0]/beta;
    object_to_track->object_cg.xv = (uint32_t) pv[0][0];
    object_to_track->object_cg.yv = (uint32_t) pv[1][0];

    // Find rate of width and height increase
    // Calculate droiw/dt and droih/dt
    object_to_track->droiwdtw = (opticflow->roiw-object_to_track->roiw_prev)/(dto*opticflow->roiw);
    object_to_track->droihdth = (opticflow->roih-object_to_track->roih_prev)/(dto*opticflow->roih);
    // As an experiment also scale x and y_gain with object size: if this works introduce scaling factor as variable

    if(elc_phase == 2 && set_x_gain){
      x_gain = pstate/p_x_ratio;
      y_gain = -1*x_gain;
    }

    // Replace divergence with dwdt
    float new_divergence = ((object_to_track->droiwdtw+object_to_track->droihdth)*div_factor)/(2*dto);
    // deal with (unlikely) fast changes in divergence:
    static const float max_div_dt = 0.20f;
    if (fabsf(new_divergence - of_landing_ctrl.divergence) > max_div_dt) {
      if (new_divergence < of_landing_ctrl.divergence) { new_divergence = of_landing_ctrl.divergence - max_div_dt; }
      else { new_divergence = of_landing_ctrl.divergence + max_div_dt; }
    }
    // low-pass filter the divergence:
    of_landing_ctrl.divergence += (new_divergence - of_landing_ctrl.divergence) * lp_factor;

		// Resetting dwdt_star
    if(object_to_track->ibvs_go2){
      object_to_track->ibvs_go2 = false;
      dwdtw_star *= 1.05;
      printf("New dwdtw_star %f\n",dwdtw_star);
    }
    // Getting velocity commands
		v_xstar = x_gain * (object_to_track->object_cg.xv-x_gstar);
		v_ystar = y_gain * (object_to_track->object_cg.yv-y_gstar);
		/*v_zstar = z_gain * ((object_to_track->droiwdt+object_to_track->droihdt)/2-dwdt_star);*/
		/*r_star = yaw_gain * (s4-alpha_star);*/

	  /***********
	  * Vertical CONTROL
	  ***********/
	  // landing indicates whether the drone is already performing a final landing procedure (flare):
	  if (landing){
	    opticflow->object_tracking = false;
	    guidance_v_set_guided_th(of_landing_ctrl.nominal_thrust);
	    struct EnuCoor_f *enu = stateGetPositionEnu_f();
      struct EnuCoor_f *venu = stateGetSpeedEnu_f();
	  }
	  else if(object_to_track->ibvs_go){
	    if (of_landing_ctrl.CONTROL_METHOD == 2) {
	      // EXPONENTIAL GAIN CONTROL:
	      static const float phase_0_set_point = 0.0f;

	      if (elc_phase == 0) {
	        // increase the gain till you start oscillating:

	        // if not yet oscillating, increase the gains:
	        // Extra condition: the divergence_history should be filled
	        if (of_landing_ctrl.elc_oscillate && opticflow->cov_div > of_landing_ctrl.cov_set_point && cov_array_filled > 0) {
	          if(bool_cov_array_filled){
	            time_at_filled = get_sys_time_float();
	            opticflow->t_0 = time_at_filled;
	            bool_cov_array_filled = false;
	          }
	          if(get_sys_time_float() - time_at_filled > of_landing_ctrl.t_transition){
	            pstate += dto * INCREASE_GAIN_PER_SECOND;
              float gain_factor = pstate / pused;
              istate *= gain_factor;
              dstate *= gain_factor;
              pused = pstate;
	          }
	        }
	        // use the divergence for control:
	        thrust_set = PID_divergence_control(phase_0_set_point, pused, istate, dstate, dto,opticflow,of_landing_ctrl);

	        // low pass filter cov div and remove outliers:
	        if (fabsf(lp_cov_div - opticflow->cov_div) < of_landing_ctrl.cov_limit) {
	          lp_cov_div = of_landing_ctrl.lp_cov_div_factor * lp_cov_div + (1 - of_landing_ctrl.lp_cov_div_factor) * opticflow->cov_div;
	        }
	        // Use cov_div instead of this added value
	        lp_cov_div = opticflow->cov_div;
	        // if oscillating, maintain a counter to see if it endures:
	        if (lp_cov_div <= of_landing_ctrl.cov_set_point) {
	          count_covdiv++;
	        } else {
	          count_covdiv = 0;
	          elc_time_start = get_sys_time_float();
	        }
	        // if the drone has been oscillating long enough, start landing:
	        if (!of_landing_ctrl.elc_oscillate ||
	            (count_covdiv > 0 && (get_sys_time_float() - elc_time_start) >= of_landing_ctrl.t_transition)) {
	          // next phase:
	          elc_phase = 1;
	          elc_time_start = get_sys_time_float();
	          opticflow->t_1 = get_sys_time_float();

	          // we don't want to oscillate, so reduce the gain:
	          elc_p_gain_start = of_landing_ctrl.reduction_factor_elc * pstate;
	          elc_i_gain_start = of_landing_ctrl.reduction_factor_elc * istate;
	          elc_d_gain_start = of_landing_ctrl.reduction_factor_elc * dstate;
	          // Offset of starting gain with minimum gain
	          elc_p_gain_off_min = elc_p_gain_start - MINIMUM_GAIN;
	          count_covdiv = 0;
	          of_landing_ctrl.sum_err = 0.0f;
	        }
	      } else if (elc_phase == 1) {
	        // control divergence to 0 with the reduced gain:
	        pstate = elc_p_gain_start;
	        pused = pstate;
	        istate = elc_i_gain_start;
	        dstate = elc_d_gain_start;

	        float t_interval = get_sys_time_float() - elc_time_start;
	        // this should not happen, but just to be sure to prevent too high gain values:
	        if (t_interval < 0) { t_interval = 0.0f; }

	        // use the divergence for control:
	        thrust_set = PID_divergence_control(phase_0_set_point, pused, istate, dstate, dto,opticflow,of_landing_ctrl);

	        // if we have been trying to hover stably again for 2 seconds and we move in the same way as the desired divergence, switch to landing:
	        if (t_interval >= 2.0f && of_landing_ctrl.divergence * of_landing_ctrl.divergence_setpoint >= 0.0f) {
	          // next phase:
	          elc_phase = 2;
	          elc_time_start = get_sys_time_float();
	          opticflow->t_2 = get_sys_time_float();
	          count_covdiv = 0;
	        }
	        to_phase_2 = true;
	      } else if (elc_phase == 2) {
          if(to_phase_2){
            p_x_ratio = pstate/x_gain;
            set_x_gain = true;
            to_phase_2 = false;
          }
          // land while exponentially decreasing the gain:
          float t_interval = get_sys_time_float() - elc_time_start;

          // this should not happen, but just to be sure to prevent too high gain values:
          if (t_interval < 0) { t_interval = 0.0f; }

          // determine the P-gain, exponentially decaying to a minimum gain:
          pstate = MINIMUM_GAIN + elc_p_gain_off_min*expf(of_landing_ctrl.divergence_setpoint * 0.05 * t_interval);

          pused = pstate;
	        // use the divergence for control:
	        thrust_set = PID_divergence_control(of_landing_ctrl.divergence_setpoint, pused, istate, dstate, dto,opticflow,of_landing_ctrl);

	        // when to make the final landing:
	        if (pstate <= of_landing_ctrl.p_land_threshold) {
	          elc_phase = 3;
	          opticflow->t_3 = get_sys_time_float();
	        }
	      } else {
	        thrust_set = final_landing_procedure(opticflow);
	      }
	      /*if(cov_array_filled>0){
          if(setting_average_thrust){
            of_landing_ctrl.nominal_thrust = (of_landing_ctrl.nominal_thrust*30.0+thrust_set)/(31.0*(float)MAX_PPRZ);
          }
          else{
            setting_average_thrust = true;
          }
        }*/
	    }
	    else if (of_landing_ctrl.CONTROL_METHOD == 1) {
        // ADAPTIVE GAIN CONTROL:
        // TODO: i-gain and d-gain are currently not adapted

        // adapt the gains according to the error in covariance:
        float error_cov = of_landing_ctrl.cov_set_point - opticflow->cov_div;
        printf("Before correction, error_cov = %f \n",error_cov);
        // limit the error_cov, which could else become very large:
        //if (error_cov > fabsf(of_landing_ctrl.cov_set_point)) { error_cov = fabsf(of_landing_ctrl.cov_set_point); }
        printf("After correction, error_cov = %f \n",error_cov);
        pstate -=  pstate * error_cov;
        printf("pstate = %f \n",pstate);
        if (pstate < MINIMUM_GAIN) { pstate = MINIMUM_GAIN; printf("At mimimum gain \n");}
        pused = pstate - (of_landing_ctrl.pgain_adaptive * pstate) * error_cov;
        printf("pused = %f \n",pused);
        // make sure pused does not become too small, nor grows too fast:
        if (pused < MINIMUM_GAIN) { pused = MINIMUM_GAIN; printf("At mimimum gain \n");}
        if (of_landing_ctrl.COV_METHOD == 1 && error_cov > 0.001) {
          printf("Half of pused, because error_cov too large \n");
          pused = 0.5 * pused;
        }

        // use the divergence for control:
        thrust_set = PID_divergence_control(of_landing_ctrl.divergence_setpoint, pused, of_landing_ctrl.igain,
                                            of_landing_ctrl.dgain, dto,opticflow,of_landing_ctrl);

        // when to make the final landing:
        if (pstate < of_landing_ctrl.p_land_threshold) {
          printf("Final landing \n");
          thrust_set = final_landing_procedure(opticflow);
        }
	    }

	    if (opticflow->in_flight) {
	      Bound(thrust_set, -1 * MAX_PPRZ, MAX_PPRZ);
	      stabilization_cmd[COMMAND_THRUST] = thrust_set;
	    }

	  }

		// Sending velocity commands
		if(object_to_track->ibvs_go){
      guidance_h_mode_changed(10);
      guidance_v_mode_changed(8);
			guidance_h_set_guided_vel(v_xstar,v_ystar);
			guidance_v_set_guided_th(thrust_set);
			struct EnuCoor_f *enu = stateGetPositionEnu_f();
      struct EnuCoor_f *venu = stateGetSpeedEnu_f();
      fprintf(opticflow->ibvs_file_logger,"%f\t%f\t%f\t%f\t%f\t%f\t%f\n",get_sys_time_float(),
                enu->x,enu->y,enu->z,venu->x,venu->y,venu->z);
			// guidance_h_set_guided_heading_rate(r_star);
      // Replace v_zadd with average of v_zarr
      //guidance_v_set_guided_vz(v_zstar);
		}
		// Update previous values for next loop
		object_to_track->prev_vision_time = object_to_track->vision_time;
		object_to_track->roiw_prev = opticflow->roiw;
    object_to_track->roih_prev = opticflow->roih;

	}
	if (opticflow->landing && opticflow->in_flight){
	  printf("Last phase \n");
    opticflow->object_tracking = false;
    guidance_v_set_guided_th(of_landing_ctrl.final_thrust);
    struct EnuCoor_f *enu = stateGetPositionEnu_f();
    struct EnuCoor_f *venu = stateGetSpeedEnu_f();
    if(enu->z<0.038){
      opticflow->in_flight = false;
      opticflow->t_landed = get_sys_time_float();
      // Write all event parameters to file
      fprintf(opticflow->ibvs_file_logger,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%f\t%f\t"
          "%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n",opticflow->t_landed,
          enu->x,enu->y,enu->z,venu->x,venu->y,venu->z,opticflow->start_pos.x,opticflow->start_pos.y
          ,opticflow->start_pos.z,opticflow->roih_init,opticflow->roiw_init
          ,opticflow->roi_center_init.x,opticflow->roi_center_init.y,opticflow->img_gray.w
          ,opticflow->img_gray.h,0.7175,of_landing_ctrl.divergence_setpoint,opticflow->t_0
          ,opticflow->t_1,opticflow->t_2,opticflow->t_3,opticflow->t_landed,opticflow->n_reinits
          ,opticflow->start_angles.phi,opticflow->start_angles.theta,opticflow->start_angles.psi);
      // Last line t_reinits
      fprintf(opticflow->ibvs_file_logger,"t_reinits");
      for(uint32_t i=0;i<opticflow->n_reinits;i++){
        fprintf(opticflow->ibvs_file_logger,"\t%f",opticflow->t_new_window[i]);
      }
      fclose(opticflow->ibvs_file_logger);
    } else {
      fprintf(opticflow->ibvs_file_logger,"%f\t%f\t%f\t%f\t%f\t%f\t%f\n",get_sys_time_float(),
          enu->x,enu->y,enu->z,venu->x,venu->y,venu->z);
    }
  }
}

/**
 * Functions copied from optical_flow_landing.c
 */

/**
 * Initialize the optical flow landing module
 */
void vertical_ctrl_module_init()
{
  // filling the of_landing_ctrl struct with default values:
  of_landing_ctrl.agl = 0.0f;
  of_landing_ctrl.agl_lp = 0.0f;
  of_landing_ctrl.vel = 0.0f;
  of_landing_ctrl.divergence_setpoint = -6.0; // For exponential gain landing, pick a negative value
  of_landing_ctrl.cov_set_point = OFL_COV_SETPOINT;
  of_landing_ctrl.cov_limit = fabsf(OFL_COV_LANDING_LIMIT);
  of_landing_ctrl.lp_const = OFL_LP_CONST;
  Bound(of_landing_ctrl.lp_const, 0.001f, 1.f);
  of_landing_ctrl.pgain = OFL_PGAIN;
  of_landing_ctrl.igain = OFL_IGAIN;
  of_landing_ctrl.dgain = OFL_DGAIN;
  of_landing_ctrl.divergence = 0.;
  of_landing_ctrl.previous_err = 0.;
  of_landing_ctrl.sum_err = 0.0f;
  of_landing_ctrl.d_err = 0.0f;
  // of_landing_ctrl.nominal_thrust = (float)guidance_v_nominal_throttle / (float)MAX_PPRZ; // copy this value from guidance
  of_landing_ctrl.nominal_thrust = 0.00007;
  of_landing_ctrl.CONTROL_METHOD = 2;
  of_landing_ctrl.COV_METHOD = OFL_COV_METHOD;
  of_landing_ctrl.delay_steps = 15;
  of_landing_ctrl.window_size = OFL_COV_WINDOW_SIZE;
  of_landing_ctrl.pgain_adaptive = OFL_PGAIN;
  of_landing_ctrl.igain_adaptive = OFL_IGAIN;
  of_landing_ctrl.dgain_adaptive = OFL_DGAIN;
  of_landing_ctrl.reduction_factor_elc =
    0.80f; // for exponential gain landing, after detecting oscillations, the gain is multiplied with this factor
  of_landing_ctrl.lp_cov_div_factor =
    0.99f; // low pass filtering cov div so that the drone is really oscillating when triggering the descent
  of_landing_ctrl.t_transition = 2.f;
  // if the gain reaches this value during an exponential landing, the drone makes the final landing.
  of_landing_ctrl.p_land_threshold = OFL_P_LAND_THRESHOLD;
  of_landing_ctrl.elc_oscillate = OFL_ELC_OSCILLATE;
  reset_all_vars();
  of_landing_ctrl.final_thrust = 0.675;
  // Subscribe to the altitude above ground level ABI messages
  //AbiBindMsgAGL(OFL_AGL_ID, &agl_ev, vertical_ctrl_agl_cb);
  // Subscribe to the optical flow estimator:
  // register telemetry:
  //AbiBindMsgOPTICAL_FLOW(OFL_OPTICAL_FLOW_ID, &optical_flow_ev, vertical_ctrl_optical_flow_cb);
  //register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_DIVERGENCE, send_divergence);
}

/**
 * Reset all variables:
 */
void reset_all_vars(void)
{
  of_landing_ctrl.agl_lp = of_landing_ctrl.agl = stateGetPositionEnu_f()->z;

  thrust_set = of_landing_ctrl.nominal_thrust * MAX_PPRZ;

  cov_div = 0.;
  normalized_thrust = of_landing_ctrl.nominal_thrust * 100;
  previous_cov_err = 0.;
  divergence_vision = 0.;
  divergence_vision_dt = 0.;
  divergence_setpoint = 0;

  vision_time = get_sys_time_float();
  prev_vision_time = vision_time;

  ind_hist = 0;
  cov_array_filled = 0;
  uint32_t i;
  for (i = 0; i < OFL_COV_WINDOW_SIZE; i++) {
    thrust_history[i] = 0;
    divergence_history[i] = 0;
  }

  landing = false;

  elc_phase = 0;
  elc_time_start = 0;
  count_covdiv = 0;
  lp_cov_div = 0.0f;

  pstate = of_landing_ctrl.pgain;
  pused = pstate;
  istate = of_landing_ctrl.igain;
  dstate = of_landing_ctrl.dgain;

  of_landing_ctrl.divergence = 0.;
  of_landing_ctrl.previous_err = 0.;
  of_landing_ctrl.sum_err = 0.;
  of_landing_ctrl.d_err = 0.;
}
/**
 * Execute a final landing procedure
 */
float final_landing_procedure(struct opticflow_t *opticflow)
{
  float thrust = of_landing_ctrl.final_thrust;
  opticflow->landing = true;

  return thrust;
}

/**
 * Set the covariance of the divergence and the thrust / past divergence
 * This funciton should only be called once per time step
 * @param[in] thrust: the current thrust value
 */
void set_cov_div(float thrust,struct opticflow_t *opticflow)
{
  // histories and cov detection:
  divergence_history[ind_hist] = of_landing_ctrl.divergence;

  normalized_thrust = thrust / (float)(MAX_PPRZ / 100.);
  thrust_history[ind_hist] = normalized_thrust;

  int ind_past = ind_hist - of_landing_ctrl.delay_steps;
  while (ind_past < 0) { ind_past += of_landing_ctrl.window_size; }
  past_divergence_history[ind_hist] = divergence_history[ind_past];

  // determine the covariance for landing detection:
  // only take covariance into account if there are enough samples in the histories:
  if (of_landing_ctrl.COV_METHOD == 0 && cov_array_filled > 0) {
    // TODO: step in landing set point causes an incorrectly perceived covariance
    opticflow->cov_div = covariance_f(thrust_history, divergence_history, of_landing_ctrl.window_size);
  } else if (of_landing_ctrl.COV_METHOD == 1 && cov_array_filled > 1) {
    // todo: delay steps should be invariant to the run frequency
    cov_div = covariance_f(past_divergence_history, divergence_history, of_landing_ctrl.window_size);
  }
  if (cov_array_filled < 2 && ind_hist + 1 == of_landing_ctrl.window_size) {
    cov_array_filled++;
    bool_cov_array_filled = true;
  }
  ind_hist = (ind_hist + 1) % of_landing_ctrl.window_size;
}

/**
 * Determine and set the thrust for constant divergence control
 * @param[out] thrust
 * @param[in] divergence_set_point: The desired divergence
 * @param[in] P: P-gain
 * @param[in] I: I-gain
 * @param[in] D: D-gain
 * @param[in] dt: time difference since last update
 */
float PID_divergence_control(float setpoint, float P, float I, float D, float dt,
    struct opticflow_t *opticflow, struct OpticalFlowLanding of_landing_ctrl)
{
  // determine the error:
  float err = setpoint - of_landing_ctrl.divergence;
  // update the controller errors:
  update_errors(err, dt);

  // PID control:
  float thrust = (of_landing_ctrl.nominal_thrust
                    + P * err
                    + I * of_landing_ctrl.sum_err
                    + D * of_landing_ctrl.d_err) * (float)MAX_PPRZ;
  // bound thrust:
  Bound(thrust, -1 * MAX_PPRZ, MAX_PPRZ);
  // update covariance
  set_cov_div(thrust,opticflow);

  return thrust;
}

/**
 * Updates the integral and differential errors for PID control and sets the previous error
 * @param[in] err: the error of the divergence and divergence setpoint
 * @param[in] dt:  time difference since last update
 */
void update_errors(float err, float dt)
{
  float lp_factor = dt / of_landing_ctrl.lp_const;
  Bound(lp_factor, 0.f, 1.f);

  // maintain the controller errors:
  of_landing_ctrl.sum_err += err;
  of_landing_ctrl.d_err += (((err - of_landing_ctrl.previous_err) / dt) - of_landing_ctrl.d_err) * lp_factor;
  of_landing_ctrl.previous_err = err;
}

// sending the divergence message to the ground station:
static void send_divergence(struct transport_tx *trans, struct link_device *dev)
{
  pprz_msg_send_DIVERGENCE(trans, dev, AC_ID,
                           &(of_landing_ctrl.divergence), &divergence_vision_dt, &normalized_thrust,
                           &cov_div, &pstate, &pused, &(of_landing_ctrl.agl));
}

// Reading from "sensors":
void vertical_ctrl_agl_cb(uint8_t sender_id UNUSED, float distance)
{
  of_landing_ctrl.agl = distance;
}

void vertical_ctrl_optical_flow_cb(uint8_t sender_id UNUSED, uint32_t stamp, int16_t flow_x UNUSED,
                                   int16_t flow_y UNUSED,
                                   int16_t flow_der_x UNUSED, int16_t flow_der_y UNUSED, float quality UNUSED, float size_divergence)
{
  divergence_vision = size_divergence;
  vision_time = ((float)stamp) / 1e6;
}

