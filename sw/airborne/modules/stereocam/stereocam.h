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

/** @file modules/stereocam/stereocam.h
 *  @brief interface to the TU Delft serial stereocam
 */

#ifndef STEREOCAM_H_
#define STEREOCAM_H_

#include "pprzlink/pprz_transport.h"
#include "math/pprz_algebra_float.h"

/* Main magneto pitot strcuture */
struct stereocam_t {
  struct link_device *device;           ///< The device which is uses for communication
  struct pprz_transport transport;      ///< The transport layer (PPRZ)
  struct FloatRMat body_to_cam;         ///< IMU to stereocam translation
  bool msg_available;                   ///< If we received a message
};

struct gate_t {
  uint8_t quality;
  float width;  // rad
  float height; // rad
  struct FloatEulers bearing;  // rad
  bool valid;
};

extern struct gate_t gate;
extern void nus_state_machine(void);

extern void stereocam_init(void);
extern void stereocam_event(void);
extern void state2stereocam(void);

extern void stereocam_parse_vel(struct FloatVect3 camera_vel, float R2);

extern float redroplet_wait;
extern uint8_t gate_count_thresh;

extern float nus_gate_heading;
extern int8_t nus_switch;
extern int16_t nus_turn_cmd, nus_climb_cmd;
extern uint8_t fit_thresh, climb_cmd_max, turn_cmd_max;
extern float nus_filter_factor;
extern bool nus_gate_detected;

#endif /* STEREOCAM_H_ */
