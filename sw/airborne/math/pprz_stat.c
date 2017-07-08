/*
 * Copyright (C) 2017 Kirk Scheper
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
 * @file pprz_array_float.c
 * @brief Array funcitons.
 *
 */

#include "pprz_stat.h"

/*********
 * Integer implementations
 *********/

/**
 * Get the mean value of an array
 * This is implemented using floats to handle scaling of all variables
 * @param[out] mean The mean value
 * @param[in] *array The array
 * @param[in] n Number of elements in the array
 */
int32_t mean_i(int32_t *array, uint32_t n_elements)
{
  // determine the mean for the vector:
  float sum = 0.;
  for (uint32_t i = 0; i < n_elements; i++) {
    sum += (float)array[i];
  }

  return (int32_t)(sum / n_elements);
}

/** Compute the variance of an array of values (integer).
 *  The variance is a measure of how far a set of numbers is spread out
 *  V(X) = E[(X-E[X])^2] = E[X^2] - E[X]^2
 *  where E[X] is the expected value of X
 *
 *  @param array pointer to an array of integer
 *  @param n_elements number of elements in the array
 *  @return variance
 */
int32_t variance_i(int32_t *array, uint32_t n_elements)
{
  return (covariance_i(array, array, n_elements));
}

/**
 * Get the covariance of two arrays
 * V(X) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]
 * where E[X] is the expected value of X
 * This is implemented using floats to handle scaling of all variables
 * @param[out] cov The covariance
 * @param[in] *arr1 The first array
 * @param[in] *arr2 The second array
 * @param[in] n Number of elements in the arrays
 */
int32_t covariance_i(int32_t *array1, int32_t *array2, uint32_t n_elements)
{
  // Determine means for each vector:
  float sumX = 0., sumY = 0., sumXY = 0;

  // Determine the covariance:
  for (uint32_t i = 0; i < n_elements; i++) {
    sumX += (float)array1[i];
    sumY += (float)array2[i];
    sumXY += (float)(array1[i]) * (float)(array2[i]);
  }

  return (int32_t)(sumXY / n_elements - sumX*sumY / (n_elements * n_elements));
}

/*********
 * Float implementations
 *********/

/**
 * Get the mean value of an array
 * @param[out] mean The mean value
 * @param[in] *array The array
 * @param[in] n Number of elements in the array
 */
float mean_f(float *array, uint32_t n_elements)
{
  // determine the mean for the vector:
  float sum = 0.;
  for (uint32_t i = 0; i < n_elements; i++) {
    sum += array[i];
  }

  return (sum / n_elements);
}

/** Compute the variance of an array of values (float).
 *  The variance is a measure of how far a set of numbers is spread out
 *  V(X) = E[(X-E[X])^2] = E[X^2] - E[X]^2
 *  where E[X] is the expected value of X
 *
 *  @param array pointer to an array of float
 *  @param n_elements number of values in the array
 *  @return variance
 */
float variance_f(float *array, uint32_t n_elements)
{
  return covariance_f(array, array, n_elements);
}

/**
 * Get the covariance of two arrays
 * V(X) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]
 * where E[X] is the expected value of X
 * @param[out] cov The covariance
 * @param[in] *arr1 The first array
 * @param[in] *arr2 The second array
 * @param[in] n Number of elements in the arrays
 */
float covariance_f(float *arr1, float *arr2, uint32_t n_elements)
{
  // Determine means for each vector:
  float sumX = 0., sumY = 0., sumXY = 0;

  // Determine the covariance:
  for (uint32_t i = 0; i < n_elements; i++) {
    sumX += arr1[i];
    sumY += arr2[i];
    sumXY += arr1[i] * arr2[i];
  }

  return (sumXY / n_elements - sumX*sumY / (n_elements * n_elements));
}
