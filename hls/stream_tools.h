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