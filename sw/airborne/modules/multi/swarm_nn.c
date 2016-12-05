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
#include "firmwares/rotorcraft/stabilization/stabilization_indi.h"

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
  struct EnuCoor_f ac2 = *acInfoGetPositionEnu_f(ti_acs[3].ac_id);

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

double layer2_out[2]; //nr_output_neurons
const double layer2_weights[9][2] = {
  { -0.3672,  -0.3484,},
  { 0.3754, -0.537,},
  { 0.886,  0.1956,},
  { -0.1652,  0.684,},
  { -0.1538,  -0.1288,},
  { 0.4614, -0.0138,},
  { 0.2718, 0.1088,},
  { 0.1112, 0.0138,},
  { -0.161, -0.072,},
};

double layer3_out[2]; //nr_output_neurons
const double layer3_weights[9][2] = {
  { -0.3672,  -0.3484,},
  { 0.3754, -0.537,},
  { 0.886,  0.1956,},
  { -0.1652,  0.684,},
  { -0.1538,  -0.1288,},
  { 0.4614, -0.0138,},
  { 0.2718, 0.1088,},
  { 0.1112, 0.0138,},
  { -0.161, -0.072,},
};
#else

#define NR_LAYERS 3
static const uint8_t nr_neurons[NR_LAYERS] = {5, 8, 2};

#define MAX_NEURONS 8
double layer_out[MAX_NEURONS+1];
double layer_in[MAX_NEURONS+1];

const double weights[NR_LAYERS-1][MAX_NEURONS][MAX_NEURONS] = {{
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
  { -0.161, -0.072,}},
};

/* controid nn
{
{ 0.0066, 0.4742, 0.8316, -0.0842,  -0.563, 0.1264, -0.7996,  -0.0832,},
{ -0.0564,  0.2682, 0.0388, 0.009,  -0.075, 0.9812, -0.7254,  0.7934,},
{ 0.1564, -0.1882,  -0.1216,  0.0216, -0.3684,  0.448,  0.833,  0.5022,},
{ -0.8944,  -0.2914,  -0.2838,  -0.0084,  -0.9336,  0.3802, 0.1388, -0.3276,},
{ 0.1306, 0.7086, -0.18,  -0.4894,  0.924,  -0.5994,  0.4542, 0.9976,},
{ -0.593, -0.711, -0.9474,  -0.3176,  1.1,  2.585,  1.776,  -0.967,},};
*/

/* controid nn
{
{ 0.5738, 0.2046,},
{ 0.3906, -0.6196,},
{ -0.4542,  0.2314,},
{ -0.5594,  0.7498,},
{ 0.3346, -0.5458,},
{ -0.2122,  0.371,},
{ 0.3254, -0.2864,},
{ 0.6944, -0.7304,},
{ -0.2692,  0.053,},};
 */

#endif

void guidance_h_module_init(void){};
void guidance_h_module_enter(void){};
void guidance_h_module_read_rc(void){};
void guidance_h_module_run(bool in_flight){
  stabilization_indi_run(FALSE, FALSE); // run yaw control
  swarm_nn_periodic();

  stabilization_cmd[COMMAND_ROLL]  = sp.x;
  stabilization_cmd[COMMAND_PITCH] = sp.y;
}

/*
 * Send my gps position to the other members of the swarm
 * Generate velocity commands based on relative position in swarm
 * using a neural network
 */
void swarm_nn_periodic(void)
{
  /* This algorithm only works if I have a gps fix. */
  //if (gps.fix == 0) {
  //  /* set default speed */
  //    autopilot_guided_move_ned(0, 0, 0, 0);
  //    return;
// }

  /* counters */
  uint8_t i, j, k;

  struct EnuCoor_f my_speed = *stateGetSpeedEnu_f();
  struct EnuCoor_f my_pos = *stateGetPositionEnu_f();
  //my_pos.x += my_speed.x * ABS(gps.tow - gps_tow_from_sys_ticks(sys_time.nb_tick)) / 1000.;
  //my_pos.y += my_speed.y * ABS(gps.tow - gps_tow_from_sys_ticks(sys_time.nb_tick)) / 1000.;
  struct EnuCoor_f ac_pos;
  struct EnuCoor_f ac_speed;

  // compute nn inputs
  rx = ry = rz = d = 0;
  for (i = 0; i < ti_acs_idx; i++) {
    if (ti_acs[i].ac_id == 0 || ti_acs[i].ac_id == AC_ID) { continue; }
    ac_pos = *acInfoGetPositionEnu_f(ti_acs[i].ac_id);
    ac_speed = *acInfoGetVelocityEnu_f(ti_acs[i].ac_id);

    // if AC not responding for too long, continue, else compute force
    float delta_t = ABS(gps.tow - acInfoGetItow(ti_acs[i].ac_id));
    if(delta_t > 5000) { continue; }

    // get distance to other with the assumption of constant velocity since last position message
    float de = ac_pos.x  - my_pos.x;// + ac_speed.x * delta_t / 1000.;
    float dn = ac_pos.y  - my_pos.y; //+ ac_speed.y * delta_t / 1000.;
    float da = ac_pos.z - my_pos.z;// + acInfoGetClimb(ti_acs[i].ac_id) * delta_t / 1000.;

    float dist2 = de * de + dn * dn;
    if (use_height) { dist2 += da * da; }

    rx += de;
    ry += dn;
    if (use_height) { rz += da; }
    d  += sqrtf(dist2);
  }

  /* set nn inputs */
  layer_in[0] = rx;
  layer_in[1] = ry;
  layer_in[2] = d;
  layer_in[3] = 0;
  layer_in[4] = 0;

#ifdef HIFI
  layer_in[3] = stateGetBodyRates_f()->p;
  layer_in[4] = stateGetBodyRates_f()->q;
  layer_in[5] = stateGetNedToBodyEulers_f()->phi;
  layer_in[6] = stateGetNedToBodyEulers_f()->theta;
  layer_in[7] = stateGetSpeedNed_f()->x;
  layer_in[8] = stateGetSpeedNed_f()->y;
#endif

  /* Compute output for hidden layer */
  for (i = 0; i < NR_LAYERS - 1; i++) {
    layer_in[nr_neurons[i]] = 1.;   // bias neuron
    for (j = 0; j < nr_neurons[i]; j++) {
      layer_out[j] = 0.;
      for (k = 0; k <= nr_neurons[i+1]; k++) {
        layer_out[j] += layer_in[k] * weights[i][k][j];
      }
      // set input for next layer
      layer_out[j] = tanh(layer_out[j]);
    }
    memcpy(layer_in, layer_out, sizeof(layer_out)); // set the input for the next layer
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
  sp.x = max_hor_speed * (float)layer_out[0];
  sp.y = max_hor_speed * (float)layer_out[1];
  if (use_height && nr_neurons[NR_LAYERS-1] > 2) {
    sp.z = max_vert_speed * (float)layer_out[2];
  }

  /* extra safety check (not strictly required) */
  BoundAbs(sp.x, max_hor_speed);
  BoundAbs(sp.y, max_hor_speed);
  BoundAbs(sp.z, max_vert_speed);

  /* set speed */
  //guidance_h_set_guided_vel(vx, vy);
  guidance_h_set_guided_vel(sp.x,sp.y); // due to error in initial evolution
#endif

}
