/*
 * Copyright (C) Kirk Scheper
 *
 * This file is part of paparazzi
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
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */
/**
 * @file "modules/multi/swarm_nn.c"
 * @author Kirk Scheper
 * Neural network based swarming algorithm
 */

#include "modules/multi/swarm_nn.h"

#include "generated/airframe.h"             // AC_ID
#include "subsystems/datalink/downlink.h"   // DefaultChannel, DefaultDevice

#include "subsystems/gps.h"                 // gps
#include "state.h"                          // my position

#include "autopilot.h"                      // autopilot_guided_move_ned
#include "guidance/guidance_h.h"
#include "math/pprz_algebra_int.h"
#include "modules/multi/traffic_info.h"     // other aircraft info

#include "stabilization.h"
#include "firmwares/rotorcraft/stabilization/stabilization_attitude.h"
#include "paparazzi.h"

#include "mcu_periph/sys_time.h"

#ifdef EXTRA_DOWNLINK_DEVICE
#include "modules/datalink/extra_pprz_dl.h"
#endif

float max_hor_speed;
float max_vert_speed;
uint8_t use_height;

#ifndef MAX_HOR_SPEED
#define MAX_HOR_SPEED 0.5
#endif

#ifndef MAX_VERT_SPEED
#define MAX_VERT_SPEED 0.5
#endif

#ifndef USE_HEIGHT
#define USE_HEIGHT 0
#endif

struct EnuCoor_f sp;
float rx, ry, rz, d;

#if PERIODIC_TELEMETRY
#include "subsystems/datalink/telemetry.h"

static void send_swarm_nn(struct transport_tx *trans, struct link_device *dev)
{
/* send log data to gs */
  struct EnuCoor_f ac1 = *acInfoGetPositionEnu_f(ti_acs[2].ac_id);
  ac1.x *= 2.49; ac1.y *= 2.49;
  struct EnuCoor_f ac2 = *acInfoGetPositionEnu_f(ti_acs[3].ac_id);
  ac2.x *= 2.49; ac2.y *= 2.49;

  struct EnuCoor_f my_enu_pos = *stateGetPositionEnu_f();

  pprz_msg_send_SWARMNN(trans, dev, AC_ID, &sp.x, &sp.y, &sp.z,
                        &rx, &ry, &rz, &d,
                        &my_enu_pos.x, &my_enu_pos.y, &my_enu_pos.z,
                        &ti_acs[2].ac_id, &ac1.x, &ac1.y,
                        &ti_acs[3].ac_id, &ac2.x, &ac2.y);
}

static void send_gps_small(struct transport_tx *trans, struct link_device *dev){
  /* First send my position to others in the swarm */

  /* The GPS messages are most likely too large to be send over either the datalink
   * The local position is an int32 and the 11 LSBs of the x and y axis are compressed into
   * a single integer. The z axis is considered unsigned and only the latter 10 LSBs are
   * used.
   */
  uint32_t multiplex_speed = (((uint32_t)(floor(DeciDegOfRad(gps.course) / 1e7) / 2)) & 0x7FF) <<
                             21; // bits 31-21 x position in cm
  multiplex_speed |= (((uint32_t)(gps.gspeed)) & 0x7FF) << 10;         // bits 20-10 y position in cm
  multiplex_speed |= (((uint32_t)(-gps.ned_vel.z)) & 0x3FF);               // bits 9-0 z position in cm

  int16_t alt = (int16_t)(gps.hmsl / 10);

#ifdef EXTRA_DOWNLINK_DEVICE
  DOWNLINK_SEND_GPS_SMALL(extra_pprz_tp, EXTRA_DOWNLINK_DEVICE, &multiplex_speed, &gps.lla_pos.lat,
                          &gps.lla_pos.lon, &alt);
#else
  pprz_msg_send_GPS_SMALL(trans, dev, AC_ID, &multiplex_speed, &gps.lla_pos.lat,
                          &gps.lla_pos.lon, &alt);
#endif
}
#endif

void swarm_nn_init(void)
{
  max_hor_speed = MAX_HOR_SPEED;
  max_vert_speed = MAX_VERT_SPEED;
  use_height = USE_HEIGHT;

  sp.x = sp.y = sp.z = 0.;
  rx = ry = rz = d = 0;

#if PERIODIC_TELEMETRY
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_SWARMNN, send_swarm_nn);
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_GPS_SMALL, send_gps_small);
#endif
}

#ifdef HIFI
#define MAX_NEURONS 9
#define NR_LAYERS 3
static const uint8_t nr_neurons[NR_LAYERS] = {6,8,8,2};

double input_layer_out[7]; //nr_input_neurons+1
double hidden_layer_out[9]; //nr_hidden_neurons+1
const double layer1_weights[7][8] = {
  { -0.249, 0.6456, 0.2576, -0.0324,  -0.1644,  0.7892, -0.313, -0.5042,},
  { 0.0736, 0.9582, 0.6542, 0.5952, -0.7244,  -0.367, -0.2034,  -0.3356,},
  { 0.1306, -0.6844,  0.8562, 0.1484, -0.4876,  -0.345, -0.611, -0.7448,},
  { 0.1306, -0.6844,  0.8562, 0.1484, -0.4876,  -0.345, -0.611, -0.7448,},
  { 0.039,  0.5906, 0.822,  0.8968, 0.2194, -0.8234,  -0.0694,  -0.3796,},
  { -0.044, -0.2248,  0.645,  -0.5442,  -0.5442,  -0.8088,  -0.67,  -0.9184,},
  { -0.142, -0.363, 2.412,  -1.586, 0.041,  -1.86,  0.443,  1.194,},
};

#else

#define NR_LAYERS 3
static const uint8_t nr_neurons[NR_LAYERS] = {5, 8, 2};

#define MAX_NEURONS 8
double layer_out[MAX_NEURONS+1];
double layer_in[MAX_NEURONS+1];

static const double weights[NR_LAYERS-1][MAX_NEURONS+1][MAX_NEURONS] = {{
{ -0.249, 0.6456, 0.2576, -0.0324,  -0.1644,  0.7892, -0.313, -0.5042,},
{ 0.0736, 0.9582, 0.6542, 0.5952, -0.7244,  -0.367, -0.2034,  -0.3356,},
{ 0.1306, -0.6844,  0.8562, 0.1484, -0.4876,  -0.345, -0.611, -0.7448,},
{ 0.039,  0.5906, 0.822,  0.8968, 0.2194, -0.8234,  -0.0694,  -0.3796,},
{ -0.044, -0.2248,  0.645,  -0.5442,  -0.5442,  -0.8088,  -0.67,  -0.9184,},
{ -0.142, -0.363, 2.412,  -1.586, 0.041,  -1.86,  0.443,  1.194,}},
{{ -0.3672,  -0.3484,},
{ 0.3754, -0.537,},
{ 0.886,  0.1956,},
{ -0.1652,  0.684,},
{ -0.1538,  -0.1288,},
{ 0.4614, -0.0138,},
{ 0.2718, 0.1088,},
{ 0.1112, 0.0138,},
{ -0.161, -0.072,}}};

/*{{
    { -0.5834,  -0.9828,  -0.0208,  -0.9592,  -0.077, 0.0192, -0.338, -0.5828,},
    { -0.166, -0.6566,  0.9078, 0.5458, 0.053,  0.4698, -0.7876,  -0.821,},
    { 0.9598, -0.4942,  -0.4448,  -0.9702,  0.4738, -0.018, -0.365, 0.239,},
    { -0.2954,  4.698,  -2.212, 0.443,  1.844,  -0.683, -0.574, -1.796,}},
    {{ 0.1708, 0.8054,},
    { -0.5764,  0.2716,},
    { -0.592, 0.7324,},
    { 0.3474, -0.0458,},
    { 0.037,  0.1778,},
    { -0.4548,  0.393,},
    { 0.415,  0.0176,},
    { -0.9162,  -0.343,},
    { -0.542, -0.344,}}};*/
#endif

void guidance_h_module_init(void){};
void guidance_h_module_enter(void){};
void guidance_h_module_read_rc(void){};

int32_t count = 0;
void guidance_h_module_run(bool in_flight)
{
  /* Check if we are in the correct AP_MODE before setting commands */
  /*if (!in_flight || autopilot_mode != AP_MODE_MODULE) {
    return;
  }*/

  /* safety check
   * if vehicle exceeds maximum bank angle switch immediately to hover mode
   * to regain stable flight
   */
  if( fabs(stateGetNedToBodyEulers_f()->phi) > STABILIZATION_ATTITUDE_SP_MAX_PHI ||
      fabs(stateGetNedToBodyEulers_f()->theta) > STABILIZATION_ATTITUDE_SP_MAX_THETA)
  {
    autopilot_mode_auto2 = AP_MODE_HOVER_Z_HOLD;
    autopilot_set_mode(AP_MODE_HOVER_Z_HOLD);
    return;
  }

  if(count++ % 5 == 0)
  {
    count = 0;
    DOWNLINK_SEND_NPS_RATE_ATTITUDE(DefaultChannel, DefaultDevice,
                            &stateGetBodyRates_f()->p, &stateGetBodyRates_f()->q,
                            &stateGetNedToBodyEulers_f()->phi, &stateGetNedToBodyEulers_f()->theta,
                            &stateGetSpeedEnu_f()->x, &stateGetSpeedEnu_f()->y);
  }

  swarm_nn_periodic();

  /* call standard stab code to control heading */
  stabilization_attitude_run(in_flight);

  // Command pitch and roll
  stabilization_cmd[COMMAND_ROLL] = sp.x;
  stabilization_cmd[COMMAND_PITCH] = sp.y;

  /* bound the result */
  BoundAbs(stabilization_cmd[COMMAND_ROLL], MAX_PPRZ);
  BoundAbs(stabilization_cmd[COMMAND_PITCH], MAX_PPRZ);
  BoundAbs(stabilization_cmd[COMMAND_YAW], MAX_PPRZ);
}

/*
 * Send my gps position to the other members of the swarm
 * Generate velocity commands based on relative position in swarm
 * using a neural network
 */
void swarm_nn_periodic(void)
{
  /* counters */
  uint8_t i, j, k;

  struct EnuCoor_f my_speed = *stateGetSpeedEnu_f();
  struct EnuCoor_f my_pos = *stateGetPositionEnu_f();
  //my_pos.x += my_speed.x * ABS(gps.tow - gps_tow_from_sys_ticks(sys_time.nb_tick)) / 1000.;
  //my_pos.y += my_speed.y * ABS(gps.tow - gps_tow_from_sys_ticks(sys_time.nb_tick)) / 1000.;
  struct EnuCoor_f ac_pos;
  struct EnuCoor_f ac_speed;
  struct EnuCoor_f* other_vel;

  // compute nn inputs
  rx = ry = rz = d = 0.;
  for (i = 0; i < ti_acs_idx; i++) {
    if (ti_acs[i].ac_id == 0 || ti_acs[i].ac_id == AC_ID) { continue; }
    ac_pos = *acInfoGetPositionEnu_f(ti_acs[i].ac_id);
    other_vel = acInfoGetVelocityEnu_f(ti_acs[i].ac_id);

    // if AC not responding for too long, continue, else compute force
    float delta_t = ABS(gps.tow - acInfoGetItow(ti_acs[i].ac_id)) / 1000.;
    //if(delta_t > 5.) { continue; }

    // get distance to other with the assumption of constant velocity since last position message
    float de = ac_pos.x*2.49 - my_pos.x;// + other_vel->x * delta_t;
    float dn = ac_pos.y*2.49 - my_pos.y;// + other_vel->y * delta_t;
    float da = ac_pos.z - my_pos.z;// + other_vel->z * delta_t;

    //printf("%d %d %f %f %f %f %f %f %f %f\n", i, ti_acs[i].ac_id, de, dn, ac_pos.x*2.49, my_pos.x, ac_pos.y*2.49, my_pos.y, ac_pos.z, my_pos.z);

    float dist2 = de * de + dn * dn;
    if (use_height) { dist2 += da * da; }

    rx += de;
    ry += dn;
    if (use_height) { rz += da; }
    d  += sqrtf(dist2);
  }

  /* set nn inputs */
  layer_out[0] = rx;
  layer_out[1] = ry;
  layer_out[2] = d;
  layer_out[3] = 0;
  layer_out[4] = 0;
#ifdef HIFI
  layer_out[3] = stateGetBodyRates_f()->p;
  layer_out[4] = stateGetBodyRates_f()->q;
  layer_out[5] = stateGetNedToBodyEulers_f()->phi;
  layer_out[6] = stateGetNedToBodyEulers_f()->theta;
  layer_out[7] = stateGetSpeedNed_f()->x;
  layer_out[8] = stateGetSpeedNed_f()->y;
#endif

  /* Compute output for hidden layer */
  for (i = 0; i < NR_LAYERS - 1; i++) {
    memcpy(layer_in, layer_out, sizeof(layer_out)); // set the input for the next layer
    layer_in[nr_neurons[i]] = 1.;   // bias neuron
    for (j = 0; j < nr_neurons[i+1]; j++) {
      layer_out[j] = 0.;
      for (k = 0; k <= nr_neurons[i]; k++) {
        layer_out[j] += layer_in[k] * weights[i][k][j];
      }
      // set input for next layer
      layer_out[j] = tanh(layer_out[j]);
    }
  }

#ifdef HIFI
  /* scale output to max speed setting */
  sp.x = 1500 * (float)layer_out[0];
  sp.y = 1500 * (float)layer_out[1];
  if (use_height && nr_neurons[NR_LAYERS-1] > 2) {
    sp.z = 1500 * (float)layer_out[2];
  }

  /* extra safety check (not strictly required) */
  BoundAbs(speed_sp.x, MAX_PPRZ);
  BoundAbs(speed_sp.y, MAX_PPRZ);
  BoundAbs(speed_sp.z, MAX_PPRZ);
#else
  /* scale output to max speed setting */
  sp.x = max_hor_speed * (float)layer_out[0];  // due to error in initial evolution
  sp.y = max_hor_speed * (float)layer_out[1];
  if (use_height && nr_neurons[NR_LAYERS-1] > 2) {
    sp.z = max_vert_speed * (float)layer_out[2];
  }

  /* extra safety check (not strictly required) */
  BoundAbs(sp.x, max_hor_speed);
  BoundAbs(sp.y, max_hor_speed);
  BoundAbs(sp.z, max_vert_speed);

  /* set speed */
  guidance_h_set_guided_vel(sp.y,sp.x); // sp in enu
#endif

}
