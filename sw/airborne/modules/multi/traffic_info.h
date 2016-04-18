/*
 * Copyright (C) Pascal Brisset, Antoine Drouin (2008), Kirk Scheper (2016)
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
 * @file "modules/mutli/traffic_info.h"
 * @author Kirk Scheper
 * Keeps track of other aircraft in airspace
 */

#ifndef TRAFFIC_INFO_H
#define TRAFFIC_INFO_H

#define NB_ACS_ID 256
#define NB_ACS 24

#include <inttypes.h>
#include <math/pprz_geodetic_int.h>

struct ac_info_ {
  uint8_t ac_id;
  struct UtmCoor_i utm;
  uint16_t course; ///< decideg (CW)
  uint16_t gspeed; ///< cm/s
  uint16_t climb;  ///< cm/s
  uint32_t itow;   ///< ms
};

extern uint8_t acs_idx;
extern uint8_t the_acs_id[NB_ACS_ID];
extern struct ac_info_ the_acs[NB_ACS];

extern void traffic_info_init(void);
extern struct ac_info_ *get_ac_info(uint8_t id);

/**
 * Set Aircraft info.
 * @param[in] id aircraft id, 0 is reserved for GCS, 1 for this aircraft (id=AC_ID)
 * @param[in] utm_east UTM east on cm
 * @param[in] utm_north UTM north in cm
 * @param[in] alt Altitude in mm above the geoid
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
 * @param[in] alt Altitude in mm above the gedoi
 * @param[in] course Course in decideg (CW)
 * @param[in] gspeed Ground speed in cm/s
 * @param[in] climb Climb rate in cm/s
 * @param[in] itow GPS time of week in ms
 */
extern void set_ac_info_lla(uint8_t id, int32_t lat, int32_t lon, int32_t alt,
                            int16_t course, uint16_t gspeed, int16_t climb, uint32_t itow);

/** Parsing functions called when an ac position message is received
*/
extern int parse_acinfo(void);

#endif