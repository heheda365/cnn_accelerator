#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>
#include "matrix_vector_unit.h"
#include "stream_tools.h"


/**
 * 不带激活的全连接层
 */
template <	unsigned IN_LEN,
            unsigned OUT_LEN,
			unsigned IN_BIT,

			unsigned W_BIT,
			unsigned M_BIT,
			unsigned SIMD,
			unsigned PE>
void linear(
	stream<ap_uint<SIMD*IN_BIT> >& in,
	const ap_uint<SIMD*W_BIT> weights[PE][(IN_LEN/SIMD)*(OUT_LEN/PE)],
	stream<ap_uint<PE*M_BIT> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    matrix_vector_unit<IN_LEN, OUT_LEN, IN_BIT, W_BIT, M_BIT, SIMD, PE, 1>(in, weights, out, reps);
	
}


/**
 * 带激活的全连接层
 */
template <	unsigned IN_LEN,
            unsigned OUT_LEN,
			unsigned IN_BIT,

            unsigned OUT_BIT,

			unsigned W_BIT,
			unsigned M_BIT,

            unsigned INC_BIT,
            unsigned BIAS_BIT,

			unsigned SIMD,
			unsigned PE>
void linear_bn_act(
	stream<ap_uint<SIMD*IN_BIT> >& in,
	const ap_uint<SIMD*W_BIT> weights[PE][(IN_LEN/SIMD)*(OUT_LEN/PE)],
    const ap_uint<INC_BIT> inc[PE][OUT_LEN/PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_LEN/PE],
	stream<ap_uint<PE*OUT_BIT> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
    matrix_vector_act_unit<IN_LEN, OUT_LEN, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, 1>(in, weights, inc, bias, out, reps);
	
}



