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

#include "generated/airframe.h"             // AC_ID, COMMAND_ROLL & COMMAND_ROLL
#include "paparazzi.h"                      // MAX_PPRZ
#include "subsystems/datalink/downlink.h"   // periodic telemetry, DefaultChannel, DefaultDevice

#include "subsystems/gps.h"                 // gps
#include "state.h"                          // my position

#include "autopilot.h"                      // autopilot_guided_move_ned and stabilization commands

#include "modules/multi/traffic_info.h"     // other aircraft info

#include "stabilization.h"

#ifdef EXTRA_DOWNLINK_DEVICE
#include "modules/datalink/extra_pprz_dl.h"
#endif

/* settings for speed control */
float max_hor_speed;
float max_vert_speed;
uint8_t use_height;

/* max pprz cmd for moment when in stability mode*/
const float max_pprz = 1500;

/* nn inputs */
float rx;
float ry;
float rz;
float d;

/* command setpoint */
struct EnuCoor_f cmd_sp;

#ifndef MAX_HOR_SPEED
#define MAX_HOR_SPEED 0.5
#endif

#ifndef MAX_VERT_SPEED
#define MAX_VERT_SPEED 0.5
#endif

#ifndef USE_HEIGHT
#define USE_HEIGHT 0
#endif

#if PERIODIC_TELEMETRY
#include "subsystems/datalink/telemetry.h"

static void send_periodic(struct transport_tx *trans, struct link_device *dev)
{
  struct EnuCoor_i* my_pos = stateGetPositionEnu_i();

  /* send log data to gs */
  struct EnuCoor_i* ac1 = acInfoGetPositionEnu_i(ti_acs[2].ac_id);
  struct EnuCoor_i* ac2 = acInfoGetPositionEnu_i(ti_acs[3].ac_id);

  int32_t tempx1 = (ac1->x - my_pos->x);
  int32_t tempy1 = (ac1->y - my_pos->y);
  int32_t tempx2 = (ac2->x - my_pos->x);
  int32_t tempy2 = (ac2->y - my_pos->y);

  struct EnuCoor_f my_enu_pos = *stateGetPositionEnu_f();

  pprz_msg_send_SWARMNN(trans, dev, AC_ID, &cmd_sp.x, &cmd_sp.y, &cmd_sp.z,
                            &rx, &ry, &rz, &d,
                            &my_enu_pos.x, &my_enu_pos.y, &my_enu_pos.z,
                            &ti_acs[2].ac_id, &tempx1, &tempy1,
                            &ti_acs[3].ac_id, &tempx2, &tempy2);
}
#endif

void swarm_nn_init(void)
{
  max_hor_speed = MAX_HOR_SPEED;
  max_vert_speed = MAX_VERT_SPEED;
  use_height = USE_HEIGHT;

  rx = 0.;
  ry = 0.;
  rz = 0.;
  d  = 0.;

  cmd_sp = (struct EnuCoor_f){0.,0.,0.};

#if PERIODIC_TELEMETRY
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_SWARMNN, send_periodic);
#endif
}

/*// the following does not include the bias neuron
static const uint8_t nr_input_neurons = 6;
static const uint8_t nr_hidden_neurons = 8;
static const uint8_t nr_output_neurons = 2;

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
};*/
/*
// the following does not include the bias neuron
static const uint8_t nr_input_neurons = 9;
static const uint8_t nr_hidden_neurons = 15;
static const uint8_t nr_output_neurons = 2;

double input_layer_out[10]; //nr_input_neurons+1 (bias node)
double hidden_layer_out[16]; //nr_hidden_neurons+1 (bias node)
const double layer1_weights[10][15] = {
    { -0.4564,  -0.325, 0.0876, -0.5824,  0.1026, 0.04, 0.449,  -0.064, -0.7184,  -0.5428,  0.7722, -0.478, -0.2256,  0.028,  0.0132,},
    { -0.019, 0.6714, -0.1124,  -0.6288,  -0.4932,  -0.0396,  0.0992, -0.0234,  0.3788, 0.627,  0.4756, 0.0074, 0.2054, 0.561,  0.0954,},
    { 0.3102, 0.9356, -0.0136,  -0.4538,  -0.5272,  -0.0122,  -0.3332,  -0.5324,  0.1706, -0.117, -0.8242,  -0.29,  -0.0796,  -0.493, -0.76,},
    { 0.4954, 0.0468, 0.9024, -0.1842,  0.7974, -0.8664,  0.3084, 0.2118, 0.3586, -0.3192,  -0.856, 0.2378, 0.8166, -0.2268,  0.442,},
    { -0.4004,  -0.0426,  0.9258, 0.7438, 0.643,  0.8836, 0.6156, -0.515, 0.487,  -0.5764,  0.376,  -0.4216,  -0.0566,  0.863,  0.4818,},
    { -0.1838,  0.1844, 0.9428, -0.568, -0.7128,  0.1972, 0.2866, -0.4458,  -0.888, -0.7918,  -0.2058,  0.193,  0.5262, 0.633,  0.1654,},
    { 0.7688, -0.5754,  0.4288, -0.1996,  -0.3224,  0.7632, -0.3612,  -0.6102,  0.0232, -0.5556,  0.3808, -0.456, 0.328,  -0.2068,  0.127,},
    { 0.8914, 0.1586, -0.9042,  0.2124, -0.0778,  -0.8654,  0.0152, 0.8406, -0.842, -0.675, 0.9286, 0.0876, -0.7352,  -0.4792,  -0.381,},
    { 0.046,  0.2038, 0.802,  -0.4192,  -0.578, -0.7634,  0.2746, -0.5496,  -0.6074,  -0.153, 0.1922, -0.0102,  0.5134, -0.6634,  0.3288,},
    { 1.192,  0.716,  0.168,  -0.769, 0.634,  -0.253, -2.414, 2.161,  -0.585, 2.85, -2.357, -3.504, 1.176,  0.259,  -0.887,},};

double layer2_out[2]; //nr_output_neurons
const double layer2_weights[16][2] = {
    { -0.023, 0.633,},
    { -0.5562,  0.2452,},
    { -0.4762,  -0.921,},
    { -0.1022,  0.4344,},
    { 0.1638, -0.1332,},
    { 0.5906, -0.8884,},
    { 0.373,  0.5414,},
    { 0.0588, 0.4544,},
    { 0.1098, -0.073,},
    { 0.338,  0.0206,},
    { -0.675, 0.7228,},
    { 0.1994, 0.1618,},
    { -0.487, -0.7804,},
    { 0.1506, 0.0144,},
    { -0.5366,  -0.6754,},
    { 0.1842, 0.265,},};*/

// the following does not include the bias neuron
static const uint8_t nr_input_neurons = 3;
static const uint8_t nr_hidden_neurons = 8;
static const uint8_t nr_output_neurons = 2;

double input_layer_out[4]; //nr_input_neurons+1 (bias node)
double hidden_layer_out[9]; //nr_hidden_neurons+1 (bias node)
const double layer1_weights[4][9] = {
    { -0.5834,  -0.9828,  -0.0208,  -0.9592,  -0.077, 0.0192, -0.338, -0.5828,},
    { -0.166, -0.6566,  0.9078, 0.5458, 0.053,  0.4698, -0.7876,  -0.821,},
    { 0.9598, -0.4942,  -0.4448,  -0.9702,  0.4738, -0.018, -0.365, 0.239,},
    { -0.2954,  4.698,  -2.212, 0.443,  1.844,  -0.683, -0.574, -1.796,},};

double layer2_out[2]; //nr_output_neurons
const double layer2_weights[9][2] = {
    { 0.1708, 0.8054,},
    { -0.5764,  0.2716,},
    { -0.592, 0.7324,},
    { 0.3474, -0.0458,},
    { 0.037,  0.1778,},
    { -0.4548,  0.393,},
    { 0.415,  0.0176,},
    { -0.9162,  -0.343,},
    { -0.542, -0.344,},};

void compute_spacial_inputs(void);
void compute_spacial_inputs(void)
{
  /* reset formation parameters */
  rx = 0.;
  ry = 0.;
  rz = 0.;
  d  = 0.;

  /* counters */
  uint8_t i;

  struct EnuCoor_f* my_pos = stateGetPositionEnu_f();

  // compute nn inputs
  for (i = 0; i < ti_acs_idx; i++) {
    if (ti_acs[i].ac_id == 0 || ti_acs[i].ac_id == AC_ID) { continue; }
    struct EnuCoor_f* other_pos = acInfoGetPositionEnu_f(ti_acs[i].ac_id);
    struct EnuCoor_f* other_vel = acInfoGetVelocityEnu_f(ti_acs[i].ac_id);

    // time since last update (s)
    float delta_t = ABS((int32_t)(gps.tow - acInfoGetItow(ti_acs[i].ac_id)))/1000.;

    // If AC not responding for too long, skip
    if(delta_t > 5.) { continue; }

    // get distance to other with the assumption of constant velocity since last position message
    float de = other_pos->x - my_pos->x + other_vel->x * delta_t;
    float dn = other_pos->y - my_pos->y + other_vel->y * delta_t;
    float da = other_pos->z - my_pos->z + other_vel->z * delta_t;

    float dist2 = de * de + dn * dn;
    if (use_height) { dist2 += da * da; }

    rx += de;
    ry += dn;
    if (use_height) { rz += da; }
    d  += sqrtf(dist2);
  }
}

void compute_nn_output(void);
void compute_nn_output(void)
{
  /* counters */
  uint8_t i, j;

  /* ensure bias nodes are set */
  input_layer_out[nr_input_neurons] = 1.;
  hidden_layer_out[nr_hidden_neurons] = 1.;

  /* Compute output for hidden layer */
  for (i = 0; i < nr_hidden_neurons; i++) {
    hidden_layer_out[i] = 0.;
    for (j = 0; j <= nr_input_neurons; j++) {
      hidden_layer_out[i] += input_layer_out[j] * layer1_weights[j][i];
    }
    hidden_layer_out[i] = tanh(hidden_layer_out[i]);
  }

  /* Compute output for output layer */
  for (i = 0; i < nr_output_neurons; i++) {
    layer2_out[i] = 0.;
    for (j = 0; j <= nr_hidden_neurons; j++) {
      layer2_out[i] += hidden_layer_out[j] * layer2_weights[j][i];
    }
    layer2_out[i] = tanh(layer2_out[i]);
  }
}

/*
 * Send my gps position to the other members of the swarm
 * Generate velocity commands based on relative position in swarm
 * using a neural network
 */
void swarm_nn_periodic(void)
{
  /* Check if we are in the correct AP_MODE before setting commands */
  if (autopilot_mode != AP_MODE_GUIDED) {
    return;
  }

  /* This algorithm only works if I have a gps fix. */
  //if (gps.fix == 0) {
  //  /* set default speed */
  //    autopilot_guided_move_ned(0, 0, 0, 0);
  //    return;
// }

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
  DOWNLINK_SEND_GPS_SMALL(DefaultChannel, DefaultDevice, &multiplex_speed, &gps.lla_pos.lat,
                          &gps.lla_pos.lon, &alt);
#endif

  /* get formation parameters */
  compute_spacial_inputs();

  /* set nn inputs */
  input_layer_out[0] = (double)rx;
  input_layer_out[1] = (double)ry;
  //input_layer_out[2] = (double)rz;
  input_layer_out[2] = (double)d;
  //input_layer_out[4] = 0; //(double)my_enu_pos.x;    // todo can add offset here later to move swarm centre
  //input_layer_out[5] = 0; //(double)my_enu_pos.y;

  /* compute nn output */
  compute_nn_output();

  /* scale output to max speed setting */
  cmd_sp.x = max_hor_speed * (float)layer2_out[0];
  cmd_sp.y = max_hor_speed * (float)layer2_out[1];
  if (use_height && nr_output_neurons > 2) {
    cmd_sp.z = max_vert_speed * (float)layer2_out[2];
  }

  /* extra safety check (not strictly required) */
  BoundAbs(cmd_sp.x, max_hor_speed);
  BoundAbs(cmd_sp.y, max_hor_speed);
  BoundAbs(cmd_sp.z, max_vert_speed);

  /* set speed */
  autopilot_guided_move_ned(cmd_sp.y, cmd_sp.x, cmd_sp.z, 0);
}

double integrator;
void guidance_h_module_init(void){};
void guidance_h_module_enter(void){
  integrator = 0;
};
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

  /* get formation parameters */
  compute_spacial_inputs();

  /* set nn inputs */
  input_layer_out[0] = (double)rx;
  input_layer_out[1] = (double)ry;
  input_layer_out[2] = (double)d;

  input_layer_out[3] = (double)stateGetBodyRates_f()->p;
  input_layer_out[4] = (double)stateGetBodyRates_f()->q;

  input_layer_out[5] = (double)stateGetNedToBodyEulers_f()->phi;
  input_layer_out[6] = (double)stateGetNedToBodyEulers_f()->theta;

  // force inputs to be in body frame
  input_layer_out[7] = (double)(stateGetSpeedEnu_f()->x * cosf(stateGetNedToBodyEulers_f()->psi) + stateGetSpeedEnu_f()->y * sinf(stateGetNedToBodyEulers_f()->psi));
  input_layer_out[8] = (double)(stateGetSpeedEnu_f()->y * cosf(stateGetNedToBodyEulers_f()->psi) + stateGetSpeedEnu_f()->x * sinf(stateGetNedToBodyEulers_f()->psi));

  if(count++ % 5 == 0)
  {
    count = 0;
    DOWNLINK_SEND_NPS_RATE_ATTITUDE(DefaultChannel, DefaultDevice,
                            &stateGetBodyRates_f()->p, &stateGetBodyRates_f()->q,
                            &stateGetNedToBodyEulers_f()->phi, &stateGetNedToBodyEulers_f()->theta,
                            &stateGetSpeedEnu_f()->x, &stateGetSpeedEnu_f()->y);
  }

  /* compute nn output */
  compute_nn_output();

  /* scale output to max speed setting */
  cmd_sp.x = max_pprz * (float)layer2_out[0];
  cmd_sp.y = max_pprz * (float)layer2_out[1];

  /* extra safety check (not strictly required) */
  BoundAbs(cmd_sp.x, max_pprz);
  BoundAbs(cmd_sp.y, max_pprz);

  /* call standard stab code to control heading */
  stabilization_attitude_run(in_flight);

  // Command pitch and roll
  stabilization_cmd[COMMAND_ROLL] = cmd_sp.x;
  stabilization_cmd[COMMAND_PITCH] = cmd_sp.y;

  /*integrator -= stateGetNedToBodyEulers_f()->psi;
  BoundAbs(integrator, 100);
  stabilization_cmd[COMMAND_YAW] = 1000 * -stateGetNedToBodyEulers_f()->psi + 10*integrator + 700 * -stateGetBodyRates_f()->r;
  */

  /* bound the result */
  BoundAbs(stabilization_cmd[COMMAND_ROLL], MAX_PPRZ);
  BoundAbs(stabilization_cmd[COMMAND_PITCH], MAX_PPRZ);
  BoundAbs(stabilization_cmd[COMMAND_YAW], MAX_PPRZ);
}
