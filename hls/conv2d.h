#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>


#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "function.h"
#include "stream_tools.h"


template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MVTU_InP,
			unsigned MVTU_OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void CONV2D_ACT_NoP(
	stream<ap_uint<Cin*Ibit> >& in, 
	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][((Cin*K*K)/MVTU_InP)*(Cout/MVTU_OutP)], 
	const ap_int<Mbit> factorA[MVTU_OutP][Cout/MVTU_OutP], 
	const ap_int<Mbit> factorB[MVTU_OutP][Cout/MVTU_OutP], 
	stream<ap_uint<Cout*Abit> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	const unsigned Dout = Din/S + (Din%S > 0);
	const unsigned IntermediateDout = S*(Dout-1) + K;
#ifdef CONV2_DEBUG
	cout << "Dout: " << Dout << endl;
	cout << "IntermediateDout: " << IntermediateDout << endl;
#endif
	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
#ifdef CONV2_DEBUG
	cout << "TopLeftPad: " << TopLeftPad << endl;
	cout << "BottomRightPad: " << BottomRightPad << endl;
#endif

	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);
#ifdef CONV2_DEBUG
	cout << "samepad_out.size(): " << samepad_out.size() << endl;
#endif

	stream<ap_uint<Cin*Ibit> > swu_out("swu_out");
	SWU_NoP<K, S, IntermediateDout, Cin, Ibit> (samepad_out, swu_out, reps);

	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<Cin*Ibit, MVTU_InP*Ibit, K*K*Dout*Dout> (swu_out, swu_out_reduced, reps);

	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
	MVAU_rowfirst<Dout*Dout, Ibit, Wbit, Mbit, Abit, Cin*K*K, Cout, MVTU_InP, MVTU_OutP, ScaleBits, FactorScaleBits>
	(swu_out_reduced, weights, factorA, factorB, out_raw, reps);
#ifdef CONV2_DEBUG
	cout << "out_raw.size(): " << out_raw.size() << endl;
#endif

	ExpandWidth<MVTU_OutP*Abit, Cout*Abit, Dout*Dout>
	(out_raw, out, reps);
}
