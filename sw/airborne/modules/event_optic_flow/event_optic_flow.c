/*
 * Copyright (C) Bas Pijnacker Hordijk
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
 * @file "modules/event_optic_flow/event_optic_flow.c"
 * @author Bas Pijnacker Hordijk
 * Event based optic flow detection and control using the Dynamic Vision Sensor (DVS).
 * Implementation is based on the following:
 * - The MAV used in this application is a customized MavTec drone on which the
 *    DVS is mounted, facing downwards.
 * - The DVS is connected through USB to an Odroid XU4 board which reads
 *    the raw event input.
 * - The Odroid processes and filters the input into 'optic flow events'.
 * - These new events are sent through UART to the Paparazzi autopilot.
 * - Real-time data is logged in Paparazzi to an SD card by the 'high speed logger' module.
 */

#include "event_optic_flow.h"
//#include "divergence_landing_control.h"

#include "mcu_periph/uart.h"
#include "mcu_periph/sys_time.h"
#include "subsystems/abi.h"
#include "subsystems/datalink/telemetry.h"
#include "math/pprz_algebra_float.h"
#include "state.h"


#ifndef DVS_PORT
#error Please define UART port connected to the DVS128 event-based camera. e.g <define name="DVS_PORT" value="uart0"/>
#endif

// Module settings
#ifndef EOF_ENABLE_DEROTATION
#define EOF_ENABLE_DEROTATION 1
#endif
PRINT_CONFIG_VAR(EOF_ENABLE_DEROTATION)

#ifndef EOF_FILTER_TIME_CONSTANT
#define EOF_FILTER_TIME_CONSTANT 0.01f
#endif
PRINT_CONFIG_VAR(EOF_FILTER_TIME_CONSTANT)

#ifndef EOF_INLIER_MAX_DIFF
#define EOF_INLIER_MAX_DIFF 0.1f
#endif
PRINT_CONFIG_VAR(EOF_INLIER_MAX_DIFF)

#ifndef EOF_DEROTATION_MOVING_AVERAGE_FACTOR
#define EOF_DEROTATION_MOVING_AVERAGE_FACTOR 0.5f
#endif
PRINT_CONFIG_VAR(EOF_DEROTATION_MOVING_AVERAGE_FACTOR)

#ifndef EOF_MIN_EVENT_RATE
#define EOF_MIN_EVENT_RATE 2000.0f
#endif
PRINT_CONFIG_VAR(EOF_MIN_EVENT_RATE)

#ifndef EOF_MIN_POSITION_VARIANCE
#define EOF_MIN_POSITION_VARIANCE 300.0f
#endif
PRINT_CONFIG_VAR(EOF_MIN_POSITION_VARIANCE)

#ifndef EOF_MIN_R2
#define EOF_MIN_R2 1.0f
#endif
PRINT_CONFIG_VAR(EOF_MIN_R2)

#ifndef EOF_DIVERGENCE_CONTROL_PGAIN
#define EOF_DIVERGENCE_CONTROL_PGAIN 1.0f
#endif
PRINT_CONFIG_VAR(EOF_DIVERGENCE_CONTROL_PGAIN)

#ifndef EOF_DIVERGENCE_CONTROL_DIV_SETPOINT
#define EOF_DIVERGENCE_CONTROL_DIV_SETPOINT 0.3f
#endif
PRINT_CONFIG_VAR(EOF_DIVERGENCE_CONTROL_DIV_SETPOINT)

#ifndef EOF_CONTROL_HOVER
#define EOF_CONTROL_HOVER 0
#endif

#ifndef EOF_CONTROL_LANDING
#define EOF_CONTROL_LANDING 0
#endif

#define IR_LEDS_SWITCH 0

// State redeclaration
struct module_state eofState;

// Algorithm parameters
uint8_t enableDerotation = EOF_ENABLE_DEROTATION;
float inlierMaxDiff = EOF_INLIER_MAX_DIFF;
float divergenceControlGainP = EOF_DIVERGENCE_CONTROL_PGAIN;
float divergenceControlSetpoint = EOF_DIVERGENCE_CONTROL_DIV_SETPOINT;
float derotationMovingAverageFactor = EOF_DEROTATION_MOVING_AVERAGE_FACTOR;

// Confidence thresholds
float minPosVariance = EOF_MIN_POSITION_VARIANCE;
float minEventRate = EOF_MIN_EVENT_RATE;
float minR2 = EOF_MIN_R2;

// Logging controls
bool irLedSwitch = IR_LEDS_SWITCH;

// Constants
const int32_t MAX_NUMBER_OF_UART_EVENTS = 100;
const float MOVING_AVERAGE_MIN_WINDOW = 5.0f;
const uint8_t EVENT_SEPARATOR = 255;
const float UART_INT16_TO_FLOAT = 10.0f;
const float LENS_DISTANCE_TO_CENTER = 0.13f; // approximate distance of lens focal length to center of OptiTrack markers
const uint32_t EVENT_BYTE_SIZE = 13; // +1 for separator
const float inactivityDecayFactor = 0.8f;
const float power = 1;

// Camera intrinsic parameters
const struct cameraIntrinsicParameters dvs128Intrinsics = {
    .principalPointX = 76.70f,
    .principalPointY = 56.93f,
    .focalLengthX = 115.0f,
    .focalLengthY = 115.0f
};

// Internal function declarations (definitions below)
enum updateStatus processUARTInput(struct flowStats* s, int32_t* N);
static void sendFlowFieldState(struct transport_tx *trans, struct link_device *dev);
int16_t uartGetInt16(struct uart_periph *p);
int32_t uartGetInt32(struct uart_periph *p);

// ----- Implementations start here -----
void event_optic_flow_init(void) {
	register_periodic_telemetry(DefaultPeriodic,
	    PPRZ_MSG_ID_EVENT_OPTIC_FLOW_EST, sendFlowFieldState);
}

void event_optic_flow_start(void) {
	// Timing
	eofState.lastTime = get_sys_time_float();
	// Reset low pass filter for rates
	eofState.ratesMA.p = 0;
	eofState.ratesMA.q = 0;
	eofState.ratesMA.r = 0;
	// (Re-)initialization
	eofState.moduleFrequency = 100.0f;
	eofState.z_NED = 0.0f;
	eofState.wxTruth = 0.0f;
	eofState.wyTruth = 0.0f;
	eofState.DTruth = 0.0f;
  struct flowField field = {0., 0., 0., 0., 0., 0.,0};
  eofState.field = field;
  flowStatsInit(&eofState.stats);
  eofState.caerInputReceived = false;
}

void event_optic_flow_periodic(void) {
  struct FloatRates *rates = stateGetBodyRates_f();
  // Moving average filtering of body rates
  eofState.ratesMA.p += (rates->p - eofState.ratesMA.p) * derotationMovingAverageFactor;
  eofState.ratesMA.q += (rates->q - eofState.ratesMA.q) * derotationMovingAverageFactor;
  eofState.ratesMA.r += (rates->r - eofState.ratesMA.r) * derotationMovingAverageFactor;

  // Obtain UART data if available
  int32_t NNew;
	enum updateStatus status = processUARTInput(&eofState.stats, &NNew);

  // Timing bookkeeping, do this after the most uncertain computations,
  // but before operations where timing info is necessary
  float currentTime = get_sys_time_float();
  float dt = currentTime - eofState.lastTime;
  eofState.moduleFrequency = 1/dt;
  eofState.lastTime = currentTime;
  eofState.stats.eventRate = (float) NNew / dt;

	if (status == UPDATE_STATS) {
		// If new events are received, recompute flow field
		// In case the flow field is ill-posed, do not update
		status = recomputeFlowField(&eofState.field, &eofState.stats,
		    inlierMaxDiff, minEventRate, minPosVariance, minR2, power, dvs128Intrinsics);
	}

	// If no update has been performed, decay flow field parameters towards zero
	if (status != UPDATE_SUCCESS) {
	  eofState.field.confidence = 0;
	}
	else {
	  // Assign timestamp to last update
	  eofState.field.t = currentTime;
	}
  // Set confidence level globally
  eofState.status = status;

  // Reset sums for next iteration
  int32_t i;
  for (i = 0; i < N_FIELD_DIRECTIONS; i++) {
    float d = 0;
    eofState.stats.sumS [i] *= d;
    eofState.stats.sumSS[i] *= d;
    eofState.stats.sumV [i] *= d;
    eofState.stats.sumVV[i] *= d;
    eofState.stats.sumSV[i] *= d;
    eofState.stats.N[i] *= d;
  }

	// Derotate flow field if enabled
//	if (enableDerotation) {
//		derotateFlowField(&eofState.field, &eofState.ratesMA);
//	}
//	else {
	  // Default: simply copy result
	  eofState.field.wxDerotated = eofState.field.wx;
	  eofState.field.wyDerotated = eofState.field.wy;
//	}

	// Update height/ground truth speeds from Optitrack
  struct NedCoor_f *pos = stateGetPositionNed_f();
  struct NedCoor_f *vel = stateGetSpeedNed_f();

//  struct FloatRMat *rot = stateGetNedToBodyRMat_f();
  struct FloatEulers *ang = stateGetNedToBodyEulers_f();
  eofState.z_NED = pos->z; // for downlink

  //TODO implement transformation below for orientation corrected ground truth
  /*struct NedCoor_f velB;
  // Transformation of speeds to body frame
  velB.x = rot->m[0][0] * vel->x + rot->m[0][1] * vel->y + rot->m[0][2] * vel->z;
  velB.y = rot->m[1][0] * vel->x + rot->m[1][1] * vel->y + rot->m[1][2] * vel->z;
  velB.z = rot->m[2][0] * vel->x + rot->m[2][1] * vel->y + rot->m[2][2] * vel->z;
  float R = -pos->z/(cosf(ang->theta)*cosf(ang->phi));*/

  eofState.wxTruth = (vel->y*cosf(ang->psi) -vel->x*sinf(ang->psi)) / (pos->z - 0.01);
  eofState.wyTruth = (vel->x*cosf(ang->psi) +vel->y*sinf(ang->psi)) / (pos->z - 0.01);
  eofState.DTruth = -vel->z / (pos->z - 0.01);

	// Set control signals
	if (EOF_CONTROL_HOVER) {

	  // Assuming a perfectly aligned downward facing camera,
	  // the camera X-axis is opposite to the body Y-axis
	  // and the Y-axis is aligned to its X-axis
	  // Further assumption: body Euler angles are small
	  float vxB = eofState.z_NED * -eofState.field.wyDerotated;
	  float vyB = eofState.z_NED * -eofState.field.wxDerotated;
	  float vzB = eofState.z_NED * eofState.field.D;
	  uint32_t timestamp = get_sys_time_usec();

	  // Update control state
	  AbiSendMsgVELOCITY_ESTIMATE(1, timestamp, vxB, vyB, vzB, 0);
	}
}

void event_optic_flow_stop(void) {
  //TODO is now present as dummy, may be removed if not required
}

int16_t uartGetInt16(struct uart_periph *p) {
  int16_t out = 0;
  out |= uart_getch(p);
  out |= uart_getch(p) << 8;
  return out;
}

int32_t uartGetInt32(struct uart_periph *p) {
  int32_t out = 0;
  out |= uart_getch(p);
  out |= uart_getch(p) << 8;
  out |= uart_getch(p) << 16;
  out |= uart_getch(p) << 24;
  return out;
}

enum updateStatus processUARTInput(struct flowStats* s, int32_t *N) {
  enum updateStatus returnStatus = UPDATE_NONE;
  // SKIPS RING BUFFER PART

  *N = 0;
  // Now scan across received data and extract events
  // Scan until read pointer is one byte behind ith the write pointer
  static bool synchronized = false;
  while(uart_char_available(&DVS_PORT) > (int32_t) EVENT_BYTE_SIZE
      && *N < 500) {
    if (synchronized) {
      // Next set of bytes contains a new event
      struct flowEvent e;
      uint8_t separator;
      int16_t x,y,u,v;
      x = uartGetInt16(&DVS_PORT);
      y = uartGetInt16(&DVS_PORT);
      e.t = uartGetInt32(&DVS_PORT);
      u = uartGetInt16(&DVS_PORT);
      v = uartGetInt16(&DVS_PORT);
      separator = uart_getch(&DVS_PORT);
      if (separator == EVENT_SEPARATOR) {
        // Full event received - we can process this further
        //TODO add timestamp checking - reject events that are outdated

        // Extract floating point position and velocity
        e.x = (float) x / UART_INT16_TO_FLOAT;
        e.y = (float) y / UART_INT16_TO_FLOAT;
        e.u = (float) u / UART_INT16_TO_FLOAT;
        e.v = (float) v / UART_INT16_TO_FLOAT;

        flowStatsUpdate(s, e, eofState.ratesMA, enableDerotation, dvs128Intrinsics);
        returnStatus = UPDATE_STATS;
        if (!eofState.caerInputReceived) {
          eofState.caerInputReceived = TRUE;
        }
        (*N)++;
      }
      else {
        // we are apparently out of sync - do not process event
        synchronized = false;
      }
    }
    else {
      // (Re)synchronize at next separator
      if (uart_getch(&DVS_PORT) == EVENT_SEPARATOR) {
        synchronized = true;
      }
    }
  }
  return returnStatus;
}

static void sendFlowFieldState(struct transport_tx *trans, struct link_device *dev) {
  float fps = eofState.moduleFrequency;
  uint8_t status = (uint8_t) eofState.status;
  float confidence = eofState.field.confidence;
  float eventRate = eofState.stats.eventRate;
  float wx = eofState.field.wx;
  float wy = eofState.field.wy;
  float D  = eofState.field.D;
  float wxDerotated = eofState.ratesMA.p;
  float wyDerotated = eofState.ratesMA.q;
//  wxDerotated = eofState.field.wxDerotated;
//  wyDerotated = eofState.field.wyDerotated;
  float wxTruth = eofState.wxTruth;
  float wyTruth = eofState.wyTruth;
  float DTruth = eofState.DTruth;

  pprz_msg_send_EVENT_OPTIC_FLOW_EST(trans, dev, AC_ID,
      &fps, &status, &confidence, &eventRate, &wx, &wy, &D, &wxDerotated, &wyDerotated,
      &wxTruth,&wyTruth,&DTruth);
}
