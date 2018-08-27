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
 * @file "modules/event_based_flow/nn.c"
 * @author Kirk Scheper
 * This module is generates a command to avoid other vehicles based on their relative gps location
 */

#include "modules/event_optic_flow/nn.h"
#include "modules/event_optic_flow/nn_weights.h"

#include "stdio.h"

#include "subsystems/gps.h"
#include "subsystems/gps/gps_datalink.h"
#include "subsystems/datalink/downlink.h"
#include "state.h"
#include "navigation.h"

#include "generated/flight_plan.h"
#include "math/pprz_geodetic_int.h"
#include "math/pprz_geodetic_double.h"

#include "generated/airframe.h"           // AC_ID
#include "subsystems/abi.h"               // rssi

#include "autopilot.h"

float input_layer_out[nr_input_neurons];
float hidden_layer_out[nr_hidden_neurons] = {0};
float layer2_out[nr_output_neurons] = {0};

static float sigmoid(float val)
{
  return 1.f / (1.f + expf(-val));
}

static float relu(float val)
{
  BoundLower(val, 0.f);
  return val;
}

static float predict_nn(float D, float Ddot, float dt)
{
  int i,j;

#if NN_TYPE == NN || NN_TYPE == RNN
  input_layer_out[0] = D + bias0[0];
  input_layer_out[1] = Ddot + bias0[1];

  float potential;
  for (i = 0; i < nr_hidden_neurons; i++)
  {
    potential = 0.f;
    for (j = 0; j < nr_input_neurons; j++)
    {
      potential += input_layer_out[j]*layer1_weights[j][i];
    }
#if NN_TYPE == RNN
    potential = hidden_layer_out[i]*recurrent_weights1[i];
#endif
    hidden_layer_out[i] = relu(potential + bias1[i]);
  }

  for (i = 0; i < nr_output_neurons; i++)
  {
    potential = 0.f;
    for (j = 0; j < nr_hidden_neurons; j++)
    {
      potential += hidden_layer_out[j]*layer2_weights[j][i];
    }
#if NN_TYPE == RNN
    potential += layer2_out[i]*recurrent_weights2[i];
#endif
    layer2_out[i] = potential + bias2[i];
  }

#elif NN_TYPE == CTRNN
  input_layer_out[0] = D + bias0[0];
  input_layer_out[1] = Ddot + bias0[1];

  float derivative;
  for (i = 0; i < nr_hidden_neurons; i++)
  {
    derivative = 0.f;
    for (j = 0; j < nr_input_neurons; j++)
    {
      derivative += sigmoid(input_layer_out[j] + bias0[j])*layer1_weights[j][i];
    }
    derivative = (derivative - hidden_layer_out[i]) / (time_const1[i] + dt);
    hidden_layer_out[i] = hidden_layer_out[i] + dt*derivative;
  }

  for (i = 0; i < nr_output_neurons; i++)
  {
    derivative = 0.f;
    for (j = 0; j < nr_hidden_neurons; j++)
    {
      derivative += sigmoid(hidden_layer_out[j] + bias1[j])*layer2_weights[j][i];
    }
    derivative = (derivative - layer2_out[i]) / (time_const2[i] + dt);
    layer2_out[i] = layer2_out[i] + dt*derivative;
  }
#endif
    return layer2_out[0];
}

static uint8_t mode;
static float start_time;
static float nominal_throttle;
float thrust_effectiveness = 0.4f;
static int nn_run(float D, float Ddot, float dt)
{
  if (autopilot_get_mode() != mode && autopilot_get_mode() == AP_MODE_GUIDED){
    guidance_v_set_guided_z(stateGetPositionNed_f()->z);
    nominal_throttle = (float)stabilization_cmd[COMMAND_THRUST]/MAX_PPRZ;
  }
  mode = autopilot_get_mode();

  if(mode != AP_MODE_GUIDED || get_sys_time_float() - start_time < 1.f){
    nominal_throttle = (nominal_throttle + (float)stabilization_cmd[COMMAND_THRUST]/MAX_PPRZ)/2.f;
    return 0;
  }
  float cmd = predict_nn(D, Ddot, dt);

  // limit commands
  Bound(cmd, -0.8, 0.5);
  guidance_v_set_guided_th(cmd*thrust_effectiveness + nominal_throttle);

  return 1;

  struct FloatVect3 accel_sp;
  uint8_t accel_sp_flag = 0b00000010; // vertical accel only

  accel_sp.z = predict_nn(D, Ddot, dt);

  // limit commands
  Bound(accel_sp.z, -0.8, 0.5);

  // AbiSendMsgACCEL_SP(ACCEL_SP_DVS_ID, accel_sp_flag, &accel_sp);

  guidance_v_set_guided_th(accel_sp.z);

  return 1;
}

static void nn_test(void)
{
  printf("%f\n", predict_nn(1.f, 1.f, 0.05));
}

/* Use optical flow estimates */
#ifndef OFL_NN_ID
#define OFL_NN_ID ABI_BROADCAST
#endif
PRINT_CONFIG_VAR(OFL_NN_ID)

static abi_event optical_flow_ev;

static void div_cb(uint8_t sender_id, uint32_t stamp, int16_t UNUSED flow_x,
    int16_t UNUSED flow_y, int16_t UNUSED flow_der_x, int16_t UNUSED flow_der_y,
    float UNUSED quality, float size_divergence)
{
  static float D = 0.f, Ddot = 0.f;
  static uint32_t last_stamp = 0;

  float dt = (stamp - last_stamp) / 1e-6;
  last_stamp = stamp;
  if (dt > 1e-5){
    Ddot = (D - size_divergence) / dt;
  }
  D = size_divergence;

  nn_run(D, Ddot, dt);
}

void nn_init(void)
{
  // bind to optial flow messages to get divergence
  //AbiBindMsgOPTICAL_FLOW(OFL_NN_ID, &optical_flow_ev, div_cb);
}

/*
 * Send my gps position to the other members of the swarm
 */
void nn_periodic(void)
{
  nn_test();
}

