#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>

/**
 *  padding 函数
 */ 
template <	unsigned IN_ROW,
			unsigned IN_COL,
            unsigned IN_CH,
			unsigned IN_BIT, 
			unsigned P>
void padding(
    // 将每一数竖看成一个元素
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps = 1)
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

	ap_uint<IN_CH*IN_BIT> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {

		for (unsigned h = 0; h < P; h++) {
			for (unsigned s = 0; s < OUT_COL; s++) {
				out.write(0);
			}
		}

		for (unsigned h = 0; h < IN_ROW; h++) {

			for ( unsigned s = 0; s < OUT_COL; s++ ) {
#pragma HLS PIPELINE II=1

				if ( (s < P) || (s >= OUT_COL-P) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}
				
				out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < P; h++) {
			for (unsigned i = 0; i < OUT_COL; i++) {
				out.write(0);
			}
		}

	}
}