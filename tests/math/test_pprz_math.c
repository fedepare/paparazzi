/*
 * Copyright (C) 2014 Felix Ruess <felix.ruess@gmail.com>
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
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

/**
 * @file test_pprz_math.c
 * @brief Tests for Paparazzi math libary.
 *
 * Using libtap to create a TAP (TestAnythingProtocol) producer:
 * https://github.com/zorgnax/libtap
 *
 */

#include "tap.h"
#include "math/pprz_algebra_int.h"
#include "math/pprz_algebra_double.h"

void test_quat_vmult(void)
{
/*

  q = [1 0 1 0; 1 0.5 0.3 0.1];
  r = [1 1 1; 2 3 4];
  n = quatrotate(q, r)
  n =
     -1.0000    1.0000    1.0000
      1.3333    5.1333    0.9333
*/

  note("Testing quaternian vector rotation\n");

  const uint8_t n_tests = 3;
  double error, max_error;
  struct DoubleVect3 v_out_d, v_error_d;
  struct DoubleVect3 v_in_d[3] = {{1.,1.,1.},
                                  {1.,1.,1.},
                                  {2.,3.,4.}};
  struct DoubleQuat q_d[3] = {{1.,0.,1.,0.},
                              {1.,0.5,0.3,0.1},
                              {1.,0.5,0.3,0.1}};
  struct DoubleVect3 v_ideal_d[3] = {{-1.,1.,1.},
                                     {0.8519,1.4741,0.3185},
                                     {1.3333,5.1333,0.9333}};

  uint8_t i;
  max_error = 0;
  for (i = 0; i < 1; i++)
  {
    double_quat_vmult(&v_out_d, &(q_d[i]), &(v_in_d[i]));

    v_error_d = v_ideal_d[i];
    VECT3_SUB(v_error_d, v_out_d);
    error = double_vect3_norm(&v_error_d);
    if (error > max_error)
    {
      max_error = error;
    }
  }

  ok((max_error < 1e-4), "double_quat_vmult max_error returned %f, %f %f %f", max_error,
      v_out_d.x, v_out_d.y, v_out_d.z);

  struct FloatVect3 v_out_f, v_error_f;
  struct FloatVect3 v_in_f[n_tests];
  struct FloatQuat q_f[n_tests];
  struct FloatVect3 v_ideal_f[n_tests];

  for (i = 0; i < n_tests; i++)
  {
    VECT3_COPY(v_in_f[i], v_in_d[i]);
    VECT3_COPY(v_ideal_f[i], v_ideal_d[i]);
    QUAT_COPY(q_f[i],q_d[i]);
  }

  max_error = 0;
  for (i = 0; i < 1; i++)
  {
    float_quat_vmult(&v_out_f, &(q_f[i]), &(v_in_f[i]));

    v_error_f = v_ideal_f[i];
    VECT3_SUB(v_error_f, v_out_f);
    error = float_vect3_norm(&v_error_f);
    if (error > max_error)
    {
      max_error = error;
    }
  }

  ok((max_error < 1e-4), "float_quat_vmult max_error returned %f, %f %f %f", max_error,
      v_out_f.x, v_out_f.y, v_out_f.z);

  struct Int32Vect3 v_out_i, v_error_i;
  struct Int32Vect3 v_in_i[n_tests];
  struct Int32Quat q_i[n_tests];
  struct Int32Vect3 v_ideal_i[n_tests];

  for (i = 0; i < n_tests; i++)
  {
    INT32_VECT3_SCALE_2(v_in_i[i],v_in_f[i],pow(2,INT32_QUAT_FRAC),1);
    INT32_VECT3_SCALE_2(v_ideal_i[i],v_ideal_f[i],pow(2,INT32_QUAT_FRAC),1);
    QUAT_BFP_OF_REAL(q_i[i],q_f[i]);
  }

  max_error = 0;
  for (i = 0; i < 1; i++)
  {
    int32_quat_vmult(&v_out_i, &(q_i[i]), &(v_in_i[i]));

    v_error_i = v_ideal_i[i];
    VECT3_SUB(v_error_i, v_out_i);
    error = QUAT1_FLOAT_OF_BFP(INT32_VECT3_NORM(v_error_i));
    if (error > max_error)
    {
      max_error = error;
    }
  }

  ok((max_error < 1e-4), "int32_quat_vmult max_error returned %f, %f %f %f", max_error,
      QUAT1_FLOAT_OF_BFP(v_out_i.x), QUAT1_FLOAT_OF_BFP(v_out_i.y), QUAT1_FLOAT_OF_BFP(v_out_i.z));
}


int main()
{
  note("running algebra math tests");
  plan(4);

  test_quat_vmult();

  /* test int32_vect2_normalize */
  struct Int32Vect2 v = {2300, -4200};
  int32_vect2_normalize(&v, 10);

  ok((v.x == 491 && v.y == -898),
     "int32_vect2_normalize([2300, -4200], 10) returned [%d, %d]", v.x, v.y);


  done_testing();
}
