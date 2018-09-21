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
 * @file "modules/event_based_flow/nn_ weights.h"
 * @author Kirk Scheper
 * This module is generates a command to avoid other vehicles based on their relative gps location
 */

#ifndef EVENT_OPTICAL_FLOW_NN_WEIGHTS_H
#define EVENT_OPTICAL_FLOW_NN_WEIGHTS_H

#include "generated/airframe.h"

#define nr_input_neurons 2
#define nr_hidden_neurons 8
#define nr_output_neurons 1

#define    NN 0
#define   RNN 1
#define CTRNN 2

#ifndef NN_TYPE
#define NN_TYPE NN
#endif

#ifndef NN_VERSION
#define NN_VERSION 6
#endif

#if NN_TYPE == NN && NN_VERSION == 6
const float layer1_weights[nr_input_neurons][nr_hidden_neurons] =
{
{3.209728077107953242e-02, 1.632315908472249832e+00, -9.275850303162087862e-02, 2.138206367862137067e-02, 5.613207455534151291e-02, 2.692163644707225995e-03, 6.277558021863660631e-02, -5.111171805297654025e-02},
{-1.076659563644622591e-01, 5.941381141163386281e-02, -3.184563413388062647e-02, 7.209445085812591497e-03, 3.452642080188372820e-02, -7.806617568648457905e-02, 2.252269522390495937e-02, -2.555575597317971681e-02},
};

const float layer2_weights[nr_hidden_neurons][nr_output_neurons] =
{
{-2.780111019495222144e-02},
{7.170405239131725228e-01},
{2.926157013493572989e-03},
{3.163348764236984662e-02},
{8.618465794807662894e-03},
{2.432231627514652023e-02},
{2.997947455695102684e-03},
{-1.429595957990887234e+00},
};

const float bias0[nr_hidden_neurons] = {-7.528236789733833145e-02, 8.435391052102265974e-02};
const float bias1[nr_hidden_neurons] = {8.061538396185422006e-02, 7.819504758164232128e-02, 3.377293499114959047e-02,
        -2.761672436022968302e-01, 4.474676980251374864e-02, 4.105031053624302179e-02, 7.988422422301533210e-02,
        6.366361901870629114e-02};
const float bias2[nr_output_neurons] = {-8.594087822079539141e-01};
#endif

#if NN_TYPE == RNN && NN_VERSION == 0
const float layer1_weights[nr_input_neurons][nr_hidden_neurons] =
{
{-0.161997,0.095075,-0.012306,-0.097525,-0.271554,-0.144204,4.973414,0.028114,},
{-0.110773,0.007927,-0.155689,-0.066771,0.140671,0.130113,0.197654,0.073744,},
};


const float layer2_weights[nr_hidden_neurons][nr_output_neurons] =
{
{-0.095112,},
{0.082485,},
{0.028576,},
{-0.158669,},
{-0.125162,},
{-0.061623,},
{0.331749,},
{0.012345,},
};

const float bias0[nr_input_neurons] = {-0.935351,-0.935351};
const float bias1[nr_hidden_neurons] = {-0.096483,-0.119108,-0.350176,-0.031242,0.002736,-0.067057,0.237690,-0.029826,};
const float bias2[nr_output_neurons] = {-0.935351,};

const float recurrent_weights0[nr_input_neurons] = {-0.214271,-0.214271};
const float recurrent_weights1[nr_hidden_neurons] = {0.007383,-0.429621,-0.072861,-0.043634,-0.040660,-0.031325,-0.075404,-0.099146,};
const float recurrent_weights2[nr_output_neurons] = {-0.214271,};

#endif

#if NN_TYPE == CTRNN && NN_VERSION == 0
const float layer1_weights[nr_input_neurons][nr_hidden_neurons] =
{
{-3.673708,-0.051521,0.127943,0.036610,-0.032603,0.009640,0.004332,-0.094477,},
{-0.189683,-0.147375,0.144771,-0.090734,-0.078714,-0.140252,0.094348,0.010413,},
};


const float layer2_weights[nr_hidden_neurons][nr_output_neurons] =
{
{-152.821708,},
{-0.008780,},
{-0.026484,},
{24.305036,},
{0.011865,},
{0.071275,},
{-0.465719,},
{-0.057278,},
};

const float bias0[nr_input_neurons] = {-0.112914,0.006333,};
const float bias1[nr_hidden_neurons] = {-0.112914,0.006333,};

const float time_const0[nr_input_neurons] = {0.000000,0.094853,};
const float time_const1[nr_hidden_neurons] = {0.000000,0.091212,0.002761,0.003066,0.024489,0.011859,0.000000,0.013640,};
const float time_const2[nr_output_neurons] = {0.000235,};

#endif

#endif  // EVENT_OPTICAL_FLOW_NN_WEIGHTS_H
