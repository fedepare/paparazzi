/*
 * flow_field_computation.c
 *
 *  Created on: Jul 18, 2016
 *      Author: bas
 */

#include "flow_field_estimation.h"
#include "mcu_periph/sys_time.h"

float DET_MIN_RESOLUTION = 1e-5;

void lowPassFilterWithThreshold(float *val, float new, float factor, float limit);

void flowStatsInit(struct flowStats *s)
{
  s->eventRate = 0.f;
  int32_t i;
  for (i = 0; i < N_FIELD_DIRECTIONS; i++) {
    s->sumS[i] = 0.f;
    s->sumSS[i] = 0.f;
    s->sumV[i] = 0.f;
    s->sumVV[i] = 0.f;
    s->sumSV[i] = 0.f;
    s->N[i] = 0.f;
    s->angles[i] = ((float) i) / N_FIELD_DIRECTIONS * M_PI;
    s->cos_angles[i] = cosf(s->angles[i]);
    s->sin_angles[i] = sinf(s->angles[i]);
  }
}

void flowStatsUpdate(struct flowStats *s, struct flowEvent e, struct FloatRates rates,
                     bool enableDerotation)
{
/*  // TODO check this! Can be converted before sending from camera
  // X,Y are defined around the camera's principal point
  float x = (e.x - intrinsics.principalPointX) * intrinsics.focalLengthX_inv;
  float y = (e.y - intrinsics.principalPointY) * intrinsics.focalLengthY_inv;
  float u = e.u * intrinsics.focalLengthX_inv;
  float v = e.v * intrinsics.focalLengthY_inv;*/

  // Find direction/index of flow
  float alpha = atan2f(e.v, e.u);
  alpha += M_PI / (2.f * N_FIELD_DIRECTIONS);
  while (alpha < 0.f) {
    alpha += M_PI;
  }
  while (alpha >= M_PI) {
    alpha -= M_PI;
  }
  int32_t a = (int32_t)(N_FIELD_DIRECTIONS * alpha / M_PI);

  // Transform flow to direction reference frame
  float S = e.x * s->cos_angles[a] + e.y * s->sin_angles[a];
  float V = e.u * s->cos_angles[a] + e.v * s->sin_angles[a];

  // Derotation in direction of flow field
  if (enableDerotation) {
    V -= -s->sin_angles[a] * (rates.p - e.y * rates.r - e.x * e.y * rates.q + e.x * e.x * rates.p)
         + s->cos_angles[a] * (rates.q - e.x * rates.r - e.x * e.y * rates.p + e.y * e.y * rates.q);
  }

  // Update flow field statistics
  s->sumS [a] += S;
  s->sumSS[a] += S * S;
  s->sumV [a] += V;
  s->sumVV[a] += V * V;
  s->sumSV[a] += S * V;
  s->N[a]++;
}

enum updateStatus recomputeFlowField(struct flowField *field, struct flowStats *s,
                                     float filterFactor, float inlierMaxDiff, float minEventRate, float minPosVariance,
                                     float minR2, float power)
{

  // Define persistant variables
  static uint32_t i;
  static float varS[N_FIELD_DIRECTIONS];
  static float c_var[N_FIELD_DIRECTIONS];

  float A[3][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
  float Y[3] = {0.f, 0.f, 0.f};
  float sumV = 0.f;
  float sumVV = 0.f;
  float sumN = 0.f;
  uint32_t nValidDirections = 0;

  // Compute rate confidence value
  float c_rate = 1.f;
  if (s->eventRate < minEventRate) {
    c_rate = s->eventRate / minEventRate;//powf(s->eventRate / minEventRate, power);
  }

  // Loop over all directions and collect total flow field information
  for (i = 0; i < N_FIELD_DIRECTIONS; i++) {
    // Skip if too few new events were added along this direction
    // If the 'moving' number of events is below 1, skip
    if (s->N[i] <= 1.f) {
      varS[i] = 0.f;
      c_var[i] = 0.f;
      continue;
    }

    // Obtain mean statistics
    float meanS  = s->sumS[i] / s->N[i];

    // Compute position variances and confidence values from mean statistics
    varS[i] = ((s->sumSS[i] / s->N[i]) - meanS * meanS);// * intrinsics.focalLengthX * intrinsics.focalLengthX;
    if (varS[i] < minPosVariance) {
      //c_var[i] *= powf(varS[i] / minPosVariance, power);
      c_var[i] = varS[i] / minPosVariance;
    } else {
      c_var[i] = 1.f;
    }

    // Weight per entry: number of elements divided by variance
    float W = c_var[i];

    // Fill in matrix entries (A is used as upper triangular matrix)
    A[0][0] += W * s->N[i] * s->cos_angles[i] * s->cos_angles[i];
    A[1][1] += W * s->N[i] * s->sin_angles[i] * s->sin_angles[i];
    A[2][2] += W * s->sumSS[i];
    A[0][1] += W * s->N[i] * s->cos_angles[i] * s->sin_angles[i];
    A[0][2] += W * s->cos_angles[i] * s->sumS[i];
    A[1][2] += W * s->sin_angles[i] * s->sumS[i];
    Y[0] += W * s->cos_angles[i] * s->sumV[i];
    Y[1] += W * s->sin_angles[i] * s->sumV[i];
    Y[2] += W * s->sumSV[i];
    sumV  += W * s->sumV[i];
    sumVV += W * s->sumVV[i];
    sumN  += W * s->N[i];
    nValidDirections++;
  }

  // Compute determinant
  float det = A[0][0] * A[1][1] * A[2][2]
              + 2 * A[0][1] * A[0][2] * A[1][2]
              - A[0][0] * A[1][2] * A[1][2]
              - A[1][1] * A[0][2] * A[0][2]
              - A[2][2] * A[0][1] * A[0][1];

  if (nValidDirections < 2 || fabsf(det) < DET_MIN_RESOLUTION) {
    return UPDATE_WARNING_SINGULAR;
  }

  // Compute matrix inverse solution
  float p[3];
  p[0] = (A[0][2] * Y[1] * A[1][2] - Y[0] * A[1][2] * A[1][2] - A[0][1] * Y[1] * A[2][2] + A[0][1] * A[1][2] * Y[2] - A[0][2] * A[1][1] * Y[2] + Y[0] * A[1][1] * A[2][2]) / det;
  p[1] = (Y[0] * A[0][2] * A[1][2] - Y[1] * A[0][2] * A[0][2] + A[0][1] * A[0][2] * Y[2] - Y[0] * A[0][1] * A[2][2] + A[0][0] * Y[1] * A[2][2] - A[0][0] * A[1][2] * Y[2]) / det;
  p[2] = (A[0][1] * A[0][2] * Y[1] - Y[2] * A[0][1] * A[0][1] + Y[0] * A[0][1] * A[1][2] - Y[0] * A[0][2] * A[1][1] - A[0][0] * Y[1] * A[1][2] + A[0][0] * A[1][1] * Y[2]) / det;

  // To check coherence in the flow field, we compute the R2 fit value
  float residualSumSquares = sumVV - (p[0] * Y[0] + p[1] * Y[1] + p[2] * Y[2]);
  float totalSumSquares = sumVV - sumV * sumV / sumN;
  float R2 = 1.f - residualSumSquares / totalSumSquares;
  float c_R2 = 1.f;
  if (R2 < minR2) {
    //c_R2 *= powf(R2 / minR2, power);
    c_R2 = R2 / minR2;
    if (c_R2 < 0.f) {
      c_R2 = 0.f;
    }
  }

  // Now update the field parameters based on the confidence values
  float c_var_max = 0.f;
  for (i = 0; i < N_FIELD_DIRECTIONS; i++) {
    if (c_var[i] > c_var_max) {
      c_var_max = c_var[i];
    }
  }

  field->confidence = c_rate * c_var_max * c_R2;
  lowPassFilterWithThreshold(&field->wx, p[0], field->confidence * filterFactor, inlierMaxDiff);
  lowPassFilterWithThreshold(&field->wy, p[1], field->confidence * filterFactor, inlierMaxDiff);
  lowPassFilterWithThreshold(&field->D, p[2], field->confidence * filterFactor, inlierMaxDiff);

  // If no problem was found, update is successful
  return UPDATE_SUCCESS;
}

void lowPassFilterWithThreshold(float *val, float new, float factor, float limit)
{
  float delta = (new - *val)*factor;
  if (delta > limit) {
    delta = limit;
  }
  if (delta < -limit) {
    delta = -limit;
  }
  *val += delta;
}
