/*
 * object_tracking.c
 *
 *  Created on: 23 Jan 2019
 *      Author: kirk
 */


void rotate_point_to_virtual(struct video_config_t *camera, int x, int y)
{
  // get position relative to frame center
  x -= camera->output_size.w / 2;
  y -= camera->output_size.h / 2;

  // TODO implement body to cam rotation
  /*
  struct FloatRMat body_to_cam_rmat;
  INT32_MAT33_ZERO(body_to_cam_rmat);
  MAT33_ELMT(body_to_cam_rmat, 0, 0) = -1 << INT32_TRIG_FRAC;
  MAT33_ELMT(body_to_cam_rmat, 1, 1) = -1 << INT32_TRIG_FRAC;
  MAT33_ELMT(body_to_cam_rmat, 2, 2) =  1 << INT32_TRIG_FRAC;

  struct Int32Vect3 target_b;
  int32_rmat_transp_vmult(&target_b, &body_to_cam_rmat, &geo.target_i);
  */

  // Body <-> LTP
  struct FloatRMat *ltp_to_body_rmat = stateGetNedToBodyRMat_f();
  float_rmat_transp_vmult(&geo.target_l, ltp_to_body_rmat, &target_b);

}
