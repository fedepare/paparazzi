/*
 * Copyright (C) Pascal Brisset, Antoine Drouin (2008), Kirk Scheper (2016)
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
 */

/**
 * @file "modules/mutli/traffic_info.h"
 * @author Kirk Scheper
 * Keeps track of other aircraft in airspace
 */

#ifndef TI_H
#define TI_H

#include <inttypes.h>

#include "math/pprz_geodetic_int.h"
#include "math/pprz_geodetic_float.h"
#include "math/pprz_geodetic_utm.h"

#include "state.h"

/**
 * @defgroup ac_info Data availability representations
 * @{
 */
#define AC_INFO_POS_UTM_I 0
#define AC_INFO_POS_LLA_I 1
#define AC_INFO_POS_ENU_I 2
#define AC_INFO_POS_UTM_F 3
#define AC_INFO_POS_LLA_F 4
#define AC_INFO_POS_ENU_F 5
#define AC_INFO_VEL_ENU_I 6
#define AC_INFO_VEL_ENU_F 7
#define AC_INFO_VEL_LOCAL_F 8

struct acInfo {
  uint8_t ac_id;
  /**
   * Holds the status bits for all acinfo position and velocity representations.
   * When the corresponding bit is set the representation
   * is already computed.
   */
  uint16_t status;

  /**
   * Position in UTM coordinates.
   * Units x,y: centimetres.
   * Units z: millimetres above MSL
   */
  struct UtmCoor_i utm_pos_i;

  /**
   * Position in Latitude, Longitude and Altitude.
   * Units lat,lon: degrees*1e7
   * Units alt: milimeters above reference ellipsoid
   */
  struct LlaCoor_i lla_pos_i;

  /**
   * Position in North East Down coordinates.
   * Units: m in BFP with #INT32_POS_FRAC
   */
  struct EnuCoor_i enu_pos_i;

  /**
   * Velocity in North East Down coordinates.
   * Units: m/s in BFP with #INT32_SPEED_FRAC
   */
  struct EnuCoor_i enu_vel_i;

  /**
   * Position in UTM coordinates.
   * Units x,y: meters.
   * Units z: meters above MSL
   */
  struct UtmCoor_f utm_pos_f;

  /**
   * Position in Latitude, Longitude and Altitude.
   * Units lat,lon: radians
   * Units alt: meters above reference ellipsoid
   */
  struct LlaCoor_f lla_pos_f;

  /**
   * Position in North East Down coordinates
   * Units: m */
  struct EnuCoor_f enu_pos_f;

  /**
   * @brief speed in North East Down coordinates
   * @details Units: m/s */
  struct EnuCoor_f enu_vel_f;

  float course;        ///< rad
  float gspeed;        ///< m/s
  float climb;         ///< m/s
  uint32_t itow;          ///< ms
};

extern uint8_t ti_acs_idx;
extern uint8_t ti_acs_id[];
extern struct acInfo ti_acs[];

extern void traffic_info_init(void);

/************************ Set functions ****************************/


/**
 * Set Aircraft info.
 * @param[in] id aircraft id, 0 is reserved for GCS, 1 for this aircraft (id=AC_ID)
 * @param[in] utm_east UTM east in cm
 * @param[in] utm_north UTM north in cm
 * @param[in] alt Altitude in m above MSL
 * @param[in] utm_zone UTM zone
 * @param[in] course Course in decideg (CW)
 * @param[in] gspeed Ground speed in m/s
 * @param[in] climb Climb rate in m/s
 * @param[in] itow GPS time of week in ms
 */
extern void set_ac_info(uint8_t id, uint32_t utm_east, uint32_t utm_north, uint32_t alt, uint8_t utm_zone,
                        uint16_t course,
                        uint16_t gspeed, uint16_t climb, uint32_t itow);

/**
 * Set Aircraft info.
 * @param[in] id aircraft id, 0 is reserved for GCS, 1 for this aircraft (id=AC_ID)
 * @param[in] lat Latitude in 1e7deg
 * @param[in] lon Longitude in 1e7deg
 * @param[in] alt Altitude in mm above MSL
 * @param[in] course Course in decideg (CW)
 * @param[in] gspeed Ground speed in cm/s
 * @param[in] climb Climb rate in cm/s
 * @param[in] itow GPS time of week in ms
 */
extern void set_ac_info_lla(uint8_t id, int32_t lat, int32_t lon, int32_t alt,
                            int16_t course, uint16_t gspeed, int16_t climb, uint32_t itow);

/** Set position from UTM coordinates (int).
* @param[in] aircraft id of aircraft info to set
* @param[in] enu_vel velocity in ENU (float)
*/
static inline void acInfoSetPositionUtm_i(uint8_t ac_id, struct UtmCoor_i *utm_pos)
{
  UTM_COPY(ti_acs[ti_acs_id[ac_id]].utm_pos_i, *utm_pos);
  /* clear bits for all position representations and only set the new one */
  ti_acs[ti_acs_id[ac_id]].status = (1 << AC_INFO_POS_UTM_I);
}

/** Set position from LLA coordinates (int).
* @param[in] aircraft id of aircraft info to set
* @param[in] enu_vel velocity in ENU (float)
*/
static inline void acInfoSetPositionLla_i(uint8_t ac_id, struct LlaCoor_i *lla_pos)
{
  LLA_COPY(ti_acs[ti_acs_id[ac_id]].lla_pos_i, *lla_pos);
  /* clear bits for all position representations and only set the new one */
  ti_acs[ti_acs_id[ac_id]].status = (1 << AC_INFO_POS_LLA_I);
}

/** Set position from UTM coordinates (float).
* @param[in] aircraft id of aircraft info to set
* @param[in] enu_vel velocity in ENU (float)
*/
static inline void acInfoSetPositionUtm_f(uint8_t ac_id, struct UtmCoor_f *utm_pos)
{
  UTM_COPY(ti_acs[ti_acs_id[ac_id]].utm_pos_f, *utm_pos);
  /* clear bits for all position representations and only set the new one */
  ti_acs[ti_acs_id[ac_id]].status = (1 << AC_INFO_POS_UTM_F);
}

/** Set position from LLA coordinates (float).
* @param[in] aircraft id of aircraft info to set
* @param[in] enu_vel velocity in ENU (float)
*/
static inline void acInfoSetPositionLla_f(uint8_t ac_id, struct LlaCoor_f *lla_pos)
{
  LLA_COPY(ti_acs[ti_acs_id[ac_id]].lla_pos_i, *lla_pos);
  /* clear bits for all position representations and only set the new one */
  ti_acs[ti_acs_id[ac_id]].status = (1 << AC_INFO_POS_LLA_F);
}

/** Set velocity from ENU coordinates (int).
* @param[in] aircraft id of aircraft info to set
* @param[in] enu_vel velocity in ENU (float)
*/
static inline void acInfoSetVelocityEnu_i(uint8_t ac_id, struct EnuCoor_i *enu_vel)
{
  VECT3_COPY(ti_acs[ti_acs_id[ac_id]].enu_vel_i, *enu_vel);
  /* clear bits for all position representations and only set the new one */
  ti_acs[ti_acs_id[ac_id]].status = (1 << AC_INFO_VEL_ENU_I);
}

/** Set velocity from ENU coordinates (float).
 * @param[in] aircraft id of aircraft info to set
 * @param[in] enu_vel velocity in ENU (float)
 */
static inline void acInfoSetPositionEnu_f(uint8_t ac_id, struct EnuCoor_f *enu_vel)
{
  VECT3_COPY(ti_acs[ti_acs_id[ac_id]].enu_vel_i, *enu_vel);
  /* clear bits for all position representations and only set the new one */
  ti_acs[ti_acs_id[ac_id]].status = (1 << AC_INFO_VEL_ENU_F);
}

/************************ Get functions ****************************/

extern void acInfoCalcPositionUtm_i(uint8_t ac_id);
extern void acInfoCalcPositionUtm_f(uint8_t ac_id);
extern void acInfoCalcPositionLla_i(uint8_t ac_id);
extern void acInfoCalcPositionLla_f(uint8_t ac_id);
extern void acInfoCalcPositionEnu_i(uint8_t ac_id);
extern void acInfoCalcPositionEnu_f(uint8_t ac_id);
extern void acInfoCalcVelocityEnu_i(uint8_t ac_id);
extern void acInfoCalcVelocityEnu_f(uint8_t ac_id);

/** Get position from UTM coordinates (int).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct UtmCoor_i *acInfoGetPositionUtm_i(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_UTM_I)) {
    acInfoCalcPositionUtm_i(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].utm_pos_i;
}

/** Get position from LLA coordinates (int).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct LlaCoor_i *acInfoGetPositionLla_i(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_UTM_I)) {
    acInfoCalcPositionLla_i(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].lla_pos_i;
}

/** Get position in local ENU coordinates (int).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct EnuCoor_i *acInfoGetPositionEnu_i(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_ENU_I)) {
    acInfoCalcPositionEnu_i(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].enu_pos_i;
}

/** Get position from UTM coordinates (float).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct UtmCoor_f *acInfoGetPositionUtm_f(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_UTM_F)) {
    acInfoCalcPositionUtm_f(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].utm_pos_f;
}

/** Get position from LLA coordinates (float).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct LlaCoor_f *acInfoGetPositionLla_f(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_UTM_F)) {
    acInfoCalcPositionLla_f(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].lla_pos_f;
}

/** Get position in local ENU coordinates (float).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct EnuCoor_f *acInfoGetPositionEnu_f(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_ENU_F)) {
    acInfoCalcPositionEnu_f(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].enu_pos_f;
}

/** Get position from ENU coordinates (int).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct EnuCoor_i *acInfoGetVelocityEnu_i(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_UTM_I)) {
    acInfoCalcVelocityEnu_i(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].enu_vel_i;
}

/** Get position from ENU coordinates (float).
 * @param[in] aircraft id of aircraft info to get
 */
static inline struct EnuCoor_f *acInfoGetVelocityEnu_f(uint8_t ac_id)
{
  if (!bit_is_set(ti_acs[ti_acs_id[ac_id]].status, AC_INFO_POS_UTM_F)) {
    acInfoCalcVelocityEnu_f(ac_id);
  }
  return &ti_acs[ti_acs_id[ac_id]].enu_vel_f;
}

/** Get vehicle course (float).
 * @param[in] aircraft id of aircraft info to get
 */
static inline float acInfoGetCourse(uint8_t ac_id)
{
  return ti_acs[ti_acs_id[ac_id]].course;
}

/** Get vehicle ground speed (float).
 * @param[in] aircraft id of aircraft info to get
 */
static inline float acInfoGetGspeed(uint8_t ac_id)
{
  return ti_acs[ti_acs_id[ac_id]].gspeed;
}

/** Get vehicle climb speed (float).
 * @param[in] aircraft id of aircraft info to get
 */
static inline float acInfoGetClimb(uint8_t ac_id)
{
  return ti_acs[ti_acs_id[ac_id]].climb;
}

/** Get time of week from lastest message (ms).
 * @param[in] aircraft id of aircraft info to get
 */
static inline uint32_t acInfoGetItow(uint8_t ac_id)
{
  return ti_acs[ti_acs_id[ac_id]].itow;
}

/** Parsing datalink and telemetry functions that contain other vehicle position
*/
extern int parse_acinfo_dl(void);

#endif