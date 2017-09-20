/*
 * Copyright (C) 2015 Kirk Scheper
 *
 * This file is part of paparazzi.
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 */

/** @file modules/stereocam/stereocam.c
 *  @brief interface to TU Delft serial stereocam
 *  Include stereocam.xml to your airframe file.
 *  Parameters STEREO_PORT, STEREO_BAUD, SEND_STEREO should be configured with stereocam.xml.
 */

#include "modules/stereocam/stereocam.h"

#include "generated/airframe.h"

#include "mcu_periph/uart.h"
#include "subsystems/datalink/telemetry.h"
#include "pprzlink/messages.h"
#include "pprzlink/intermcu_msg.h"

#include "mcu_periph/sys_time.h"
#include "subsystems/abi.h"

#include "stereocam_follow_me/follow_me.h"

#include "firmwares/rotorcraft/stabilization/stabilization_attitude_rc_setpoint.h"
#include "subsystems/radio_control.h"
#include "modules/droplet/droplet.h"


// forward received image to ground station
#ifndef FORWARD_IMAGE_DATA
#define FORWARD_IMAGE_DATA FALSE
#endif


/* This defines the location of the stereocamera with respect to the body fixed coordinates.
 *
 *    Coordinate system stereocam (image coordinates)
 *    z      x
 * (( * ))----->
 *    |                       * = arrow pointed into the frame away from you
 *    | y
 *    V
 *
 * The conversion order in euler angles is psi (yaw) -> theta (pitch) -> phi (roll)
 *
 * Standard rotations: MAV NED body to stereocam in Deg:
 * - facing forward:   90 -> 0 -> 90
 * - facing backward: -90 -> 0 -> 90
 * - facing downward:  90 -> 0 -> 0
 */

// general stereocam definitions
#if !defined(STEREO_BODY_TO_STEREO_PHI) || !defined(STEREO_BODY_TO_STEREO_THETA) || !defined(STEREO_BODY_TO_STEREO_PSI)
#warning "STEREO_BODY_TO_STEREO_XXX not defined. Using default Euler rotation angles (0,0,0)"
#endif

#ifndef STEREO_BODY_TO_STEREO_PHI
#define STEREO_BODY_TO_STEREO_PHI 0
#endif

#ifndef STEREO_BODY_TO_STEREO_THETA
#define STEREO_BODY_TO_STEREO_THETA 0
#endif

#ifndef STEREO_BODY_TO_STEREO_PSI
#define STEREO_BODY_TO_STEREO_PSI 0
#endif

struct stereocam_t stereocam = {
  .device = (&((UART_LINK).device)),
  .msg_available = false
};
static uint8_t stereocam_msg_buf[256]  __attribute__((aligned));   ///< The message buffer for the stereocamera

// incoming messages definitions
#ifndef STEREOCAM2STATE_SENDER_ID
#define STEREOCAM2STATE_SENDER_ID ABI_BROADCAST
#endif

#ifndef STEREOCAM_USE_MEDIAN_FILTER
#define STEREOCAM_USE_MEDIAN_FILTER 1
#endif

#include "filters/median_filter.h"
struct MedianFilter3Float medianfilter_vel;
struct MedianFilterFloat medianfilter_noise;

struct gate_t gate;

float redroplet_wait;
uint8_t gate_count_thresh;

int16_t nus_turn_cmd;
int16_t nus_climb_cmd;
float nus_gate_heading; // gate heading relative to the current heading, in degrees
int8_t nus_switch;
uint8_t gate_count;

uint8_t fit_thresh; // min fitness that is still considered as correct detection
uint8_t turn_cmd_max; // percentage of MAX_PPRZ
uint8_t climb_cmd_max; // percentage of MAX_PPRZ
float nus_filter_factor; // complementary filter setting

uint8_t fps;

uint32_t disparities_high, processed_pixels, hist_obs_sum, count_disps_left, count_disps_right;

static void send_stereo_data(struct transport_tx *trans, struct link_device *dev)
 {
  pprz_msg_send_STEREO_DATA(trans, dev, AC_ID,
                         &gate.bearing.psi, &gate.bearing.theta, &gate.width, &gate.quality,
                         &nus_turn_cmd, &nus_climb_cmd, &nus_gate_heading, &gate_count, &droplet_active,
                         &fps);
  pprz_msg_send_STEREO_LOW_TEXTURE(trans, dev, AC_ID,
                       &disparities_high, &processed_pixels, &hist_obs_sum,
                       &count_disps_left, &count_disps_right);
}

void stereocam_init(void)
{

  redroplet_wait = 5.;
  gate_count_thresh = 2;

  nus_turn_cmd = 0;
  nus_climb_cmd = 0;
  nus_gate_heading = 0.f; // gate heading relative to the current heading, in degrees
  nus_switch = 0;
  gate_count = 0;

  fit_thresh = 14; // min fitness that is still considered as correct detection
  turn_cmd_max = 50; // percentage of MAX_PPRZ
  climb_cmd_max = 10; // percentage of MAX_PPRZ
  nus_filter_factor = 0.3; // complementary filter setting

  fps = 0;

  struct FloatEulers euler;
  euler.phi = STEREO_BODY_TO_STEREO_PHI;
  euler.theta = STEREO_BODY_TO_STEREO_THETA;
  euler.psi = STEREO_BODY_TO_STEREO_PSI;

  float_rmat_of_eulers(&stereocam.body_to_cam, &euler);

  // Initialize transport protocol
  pprz_transport_init(&stereocam.transport);

  InitMedianFilterVect3Float(medianfilter_vel, MEDIAN_DEFAULT_SIZE);
  init_median_filter_f(&medianfilter_noise, MEDIAN_DEFAULT_SIZE);

  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_STEREO_DATA, send_stereo_data);
  //register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_OPTIC_FLOW_EST, send_opticflow);
}

void stereocam_parse_vel(struct FloatVect3 camera_vel, float R2)
{
  uint32_t now_ts = get_sys_time_usec();
  float noise = 1.5*(1.1 - R2);

  // Rotate camera frame to body frame
  static struct FloatVect3 body_vel;
  float_rmat_transp_vmult(&body_vel, &stereocam.body_to_cam, &camera_vel);

  //todo make setting
  if (STEREOCAM_USE_MEDIAN_FILTER) {
    // Use a slight median filter to filter out the large outliers before sending it to state
    UpdateMedianFilterVect3Float(medianfilter_vel, body_vel);
    update_median_filter_f(&medianfilter_noise, noise);
  }

  DOWNLINK_SEND_IMU_MAG(DOWNLINK_TRANSPORT, DOWNLINK_DEVICE, &body_vel.x, &body_vel.y, &body_vel.z);

  //if(stateGetPositionEnu_f()->z > 0.4 && body_vel.x < 2.f && body_vel.y < 2.f)
  //{
    //Send velocities to state
    AbiSendMsgVELOCITY_ESTIMATE(STEREOCAM2STATE_SENDER_ID, now_ts,
                                body_vel.x,
                                body_vel.y,
                                body_vel.z,
                                noise
                               );
  //}

  // todo activate this after changing optical flow message to be dimentionless instead of in pixels
  /*
  static struct FloatVect3 camera_flow;

  float avg_dist = (float)DL_STEREOCAM_VELOCITY_avg_dist(stereocam_msg_buf)/res;

  camera_flow.x = (float)DL_STEREOCAM_VELOCITY_velx(stereocam_msg_buf)/DL_STEREOCAM_VELOCITY_avg_dist(stereocam_msg_buf);
  camera_flow.y = (float)DL_STEREOCAM_VELOCITY_vely(stereocam_msg_buf)/DL_STEREOCAM_VELOCITY_avg_dist(stereocam_msg_buf);
  camera_flow.z = (float)DL_STEREOCAM_VELOCITY_velz(stereocam_msg_buf)/DL_STEREOCAM_VELOCITY_avg_dist(stereocam_msg_buf);

  struct FloatVect3 body_flow;
  float_rmat_transp_vmult(&body_flow, &body_to_stereocam, &camera_flow);

  AbiSendMsgOPTICAL_FLOW(STEREOCAM2STATE_SENDER_ID, now_ts,
                              body_flow.x,
                              body_flow.y,
                              body_flow.z,
                              quality,
                              body_flow.z,
                              avg_dist
                             );
  */
}

/* Parse the InterMCU message */
static void stereocam_parse_msg(void)
{
  uint32_t now_ts = get_sys_time_usec();

  /* Parse the stereoboard message */
  uint8_t msg_id = stereocam_msg_buf[1];
  switch (msg_id) {

  case DL_STEREOCAM_VELOCITY: {
    static struct FloatVect3 camera_vel;

    float res = (float)DL_STEREOCAM_VELOCITY_resolution(stereocam_msg_buf);

    camera_vel.x = (float)DL_STEREOCAM_VELOCITY_velx(stereocam_msg_buf)/res;
    camera_vel.y = (float)DL_STEREOCAM_VELOCITY_vely(stereocam_msg_buf)/res;
    camera_vel.z = (float)DL_STEREOCAM_VELOCITY_velz(stereocam_msg_buf)/res;

    float R2 = (float)DL_STEREOCAM_VELOCITY_vRMS(stereocam_msg_buf)/res;

    stereocam_parse_vel(camera_vel, R2);

    fps = DL_STEREOCAM_VELOCITY_dt(stereocam_msg_buf);
    break;
  }

  case DL_STEREOCAM_ARRAY: {
#if FORWARD_IMAGE_DATA
    // forward image to ground station
    uint8_t type = DL_STEREOCAM_ARRAY_type(stereocam_msg_buf);
    uint8_t w = DL_STEREOCAM_ARRAY_width(stereocam_msg_buf);
    uint8_t h = DL_STEREOCAM_ARRAY_height(stereocam_msg_buf);
    uint8_t nb = DL_STEREOCAM_ARRAY_package_nb(stereocam_msg_buf);
    uint8_t l = DL_STEREOCAM_ARRAY_image_data_length(stereocam_msg_buf);

    DOWNLINK_SEND_STEREO_IMG(DefaultChannel, DefaultDevice, &type, &w, &h, &nb,
        l, DL_STEREOCAM_ARRAY_image_data(stereocam_msg_buf));
#endif
    break;
  }

#ifdef STEREOCAM_FOLLOWME
  // todo is follow me still used?
  case DL_STEREOCAM_FOLLOW_ME: {
    follow_me( DL_STEREOCAM_FOLLOW_ME_headingToFollow(stereocam_msg_buf),
               DL_STEREOCAM_FOLLOW_ME_heightObject(stereocam_msg_buf),
               DL_STEREOCAM_FOLLOW_ME_distanceToObject(stereocam_msg_buf));
    break;
  }
#endif

  case DL_STEREOCAM_GATE: {
    gate.quality = DL_STEREOCAM_GATE_quality(stereocam_msg_buf);
    gate.width = DL_STEREOCAM_GATE_width(stereocam_msg_buf);
    gate.height = DL_STEREOCAM_GATE_hieght(stereocam_msg_buf);
    //float d = DL_STEREOCAM_GATE_depth(stereocam_msg_buf);

    // rotate angles to body frame
    static struct FloatEulers gate_bearing_cam;
    gate_bearing_cam.phi = DL_STEREOCAM_GATE_phi(stereocam_msg_buf);
    gate_bearing_cam.theta = DL_STEREOCAM_GATE_theta(stereocam_msg_buf);
    gate_bearing_cam.psi = 0;

    float_rmat_transp_mult(&gate.bearing, &stereocam.body_to_cam, &gate_bearing_cam);
    gate.valid = true;
    break;
  }

  case DL_STEREOCAM_DROPLET: {
    disparities_high = DL_STEREOCAM_DROPLET_disparities_high(stereocam_msg_buf);
    processed_pixels = DL_STEREOCAM_DROPLET_processed_pixels(stereocam_msg_buf);
    hist_obs_sum = DL_STEREOCAM_DROPLET_hist_obs_sum(stereocam_msg_buf);
    count_disps_left = DL_STEREOCAM_DROPLET_count_disps_left(stereocam_msg_buf);
    count_disps_right = DL_STEREOCAM_DROPLET_count_disps_right(stereocam_msg_buf);

    run_droplet_low_texture(disparities_high, processed_pixels, hist_obs_sum, count_disps_left, count_disps_right);
    break;
  }

  default:
    break;
  }
}

/* We need to wait for incoming messages */
void stereocam_event(void) {
  // Check if we got some message from the stereocamera
  pprz_check_and_parse(stereocam.device, &stereocam.transport, stereocam_msg_buf, &stereocam.msg_available);

  // If we have a message we should parse it
  if (stereocam.msg_available) {
    stereocam_parse_msg();
    stereocam.msg_available = false;
  }
}

/* Send state to camera to facilitate derotation
 *
 */
void state2stereocam(void)
{
  // rotate body angles to camera reference frame
  static struct FloatEulers cam_angles;
  float_rmat_mult(&cam_angles, &stereocam.body_to_cam, stateGetNedToBodyEulers_f());

  float agl = 0;//stateGetAgl);
  pprz_msg_send_STEREOCAM_STATE(&(stereocam.transport.trans_tx), stereocam.device,
      AC_ID, &(cam_angles.phi), &(cam_angles.theta), &(cam_angles.psi), &agl);
}


void nus_state_machine(void)
{
  static float gate_time = 0.f;
  static enum nus_state_t {WINDOW_TRACKING, WINDOW_FLY_THROUGH, ROOM_EXPLORATION} nus_state = ROOM_EXPLORATION;

  if (radio_control.values[5] < 0)  // this should be ELEV D/R
  {
    nus_climb_cmd = 0;
    nus_turn_cmd = 0;
    droplet_active = 0;
    nus_switch = 0;
    return;
  } else if (radio_control.values[5] > 0 && !nus_switch)
  {
    // reset all parameters to initial state
    droplet_turn_direction = INIT_TURN_DIR;
    gate_count = 0;

    nus_gate_heading = 0.f;
    nus_state = ROOM_EXPLORATION;
    droplet_active = 1;

    nus_switch = 1;
  }

  

  // switch(nus_state)
  // {
  //   case WINDOW_TRACKING:
  //     if (gate.valid && gate.quality > 15)
  //     {
  //       /* simple filter */
  //       // todo add dt here...
  //       nus_gate_heading = (1 - nus_filter_factor) * nus_gate_heading
  //           + gate.bearing.psi * nus_filter_factor;

  //       /* add the gate heading to the current heading */
  //       stab_att_sp_euler.psi = ANGLE_BFP_OF_REAL(nus_gate_heading) + stabilization_attitude_get_heading_i();
  //       INT32_ANGLE_NORMALIZE(stab_att_sp_euler.psi);
  //       gate_time = get_sys_time_float();
  //     } else if (get_sys_time_float() > gate_time + redroplet_wait)
  //     {
  //       stab_att_sp_euler.psi += ANGLE_BFP_OF_REAL(RadOfDeg((float)droplet_turn_direction * 60.f));
  //       INT32_ANGLE_NORMALIZE(stab_att_sp_euler.psi);
  //       gate_time = get_sys_time_float();
  //       nus_state = WINDOW_FLY_THROUGH;
  //     }
  //     break;
  //   case WINDOW_FLY_THROUGH:
  //     // wait to fly-through window
  //     if (get_sys_time_float() > gate_time + 2.f)
  //     {
  //       droplet_turn_direction = -droplet_turn_direction;         // reverse droplet direction
  //       gate_count = 0;                           // reset gate counter

  //       nus_climb_cmd = 0;
  //       nus_gate_heading = 0.f;

  //       nus_state = ROOM_EXPLORATION;
  //     }
  //     break;
  //   case ROOM_EXPLORATION:
  //     droplet_active = 1;
  //     if (gate.valid){
  //       if (gate.quality > fit_thresh) // valid gate detection
  //       {
  //         stab_att_sp_euler.psi = ANGLE_BFP_OF_REAL(gate.bearing.psi) + stabilization_attitude_get_heading_i();
  //         if(++gate_count >= gate_count_thresh)
  //         {
  //           // temporarily deactivate droplet
  //           droplet_active = 0;
  //           nus_turn_cmd = 0;
  //           gate_time = get_sys_time_float();
  //           nus_state = WINDOW_TRACKING;
  //         }
  //       } else if(gate_count > 0){
  //         gate_count--;
  //       }
  //       gate.valid = false;
  //     }
  //     break;

  //   default:
  //     break;
  // }
}
