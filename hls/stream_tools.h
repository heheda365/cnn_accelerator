#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#include <assert.h>


template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned NumLines>
void ReduceWidth(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned reps = 1)
{
	static_assert( InStreamW%OutStreamW == 0, "For ReduceWidth, InStreamW mod OutStreamW is not 0" );

	const unsigned parts = InStreamW/OutStreamW;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
#pragma HLS PIPELINE II=InStreamW/OutStreamW

		ap_uint<InStreamW> temp_in = in.read();
		for (unsigned p = 0; p < parts; p++) {

			ap_uint<OutStreamW> temp_out = temp_in(OutStreamW-1, 0);
			out.write( temp_out );
			temp_in = temp_in >> OutStreamW;
		}
	}
}

template <	unsigned InStreamW,
			unsigned OutStreamW,
			unsigned NumLines>
void ExpandWidth(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned reps = 1)
{
	static_assert( OutStreamW%InStreamW == 0, "For ExpandWidth, OutStreamW mod InStreamW is not 0" );

	const unsigned parts = OutStreamW/InStreamW;
	ap_uint<OutStreamW> buffer;

	for (unsigned rep = 0; rep < reps*NumLines; rep++) {
		
		for (unsigned p = 0; p < parts; p++) {
#pragma HLS PIPELINE II=1
			ap_uint<InStreamW> temp = in.read();
			buffer( (p+1)*InStreamW-1, p*InStreamW ) = temp;
		}
		out.write(buffer);
		
	}
}

/**
 *  
 * 
 */
template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned IN_NUMS>
void adjust_width(
	stream<ap_uint<IN_BIT> > & in,
	stream<ap_uint<OUT_BIT> > & out,
	const unsigned reps = 1)
{	
	static_assert( !(IN_BIT > OUT_BIT && IN_BIT%OUT_BIT != 0), "For ReduceWidth, InStreamW mod OutStreamW is not 0" );
	static_assert( !(IN_BIT < OUT_BIT && OUT_BIT%IN_BIT != 0), "For ExpandWidth, OutStreamW mod InStreamW is not 0" );

	if (IN_BIT > OUT_BIT) {
		// 减小位宽
		const unsigned PARTS = IN_BIT/OUT_BIT;

		for (unsigned rep = 0; rep < reps*IN_NUMS; rep++) {
#pragma HLS PIPELINE II=InStreamW/OutStreamW

			ap_uint<IN_BIT> temp_in = in.read();
			for (unsigned p = 0; p < PARTS; p++) {

				ap_uint<OUT_BIT> temp_out = temp_in(OUT_BIT-1, 0);
				out.write( temp_out );
				temp_in = temp_in >> OUT_BIT;
			}
		}

	} else if (IN_BIT == OUT_BIT) {
		// 位宽不变
		// straight-through copy
    	for (unsigned int i = 0; i < IN_NUMS * reps; i++) {
#pragma HLS PIPELINE II=1
      		ap_uint<IN_BIT> e = in.read();
      		out.write(e);
    	}
	} else {
		// 增大位宽
		const unsigned PARTS = OUT_BIT/IN_BIT;
		const unsigned OUT_NUMS = IN_NUMS / PARTS;
		ap_uint<OUT_BIT> buffer;

		for (unsigned rep = 0; rep < reps*OUT_NUMS; rep++) {
			
			for (unsigned p = 0; p < PARTS; p++) {
#pragma HLS PIPELINE II=1
				ap_uint<IN_BIT> temp = in.read();
				buffer( (p+1)*IN_BIT-1, p*IN_BIT ) = temp;
			}
			out.write(buffer);
		
		}
	}
}