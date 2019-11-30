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


// template <	unsigned K,
// 			unsigned S,
// 			unsigned Din,
// 			unsigned Cin,
// 			unsigned Cout,
// 			unsigned Ibit,
// 			unsigned Wbit,
// 			unsigned Mbit,
// 			unsigned Abit,
// 			unsigned MVTU_InP,
// 			unsigned MVTU_OutP,
// 			unsigned ScaleBits,
// 			unsigned FactorScaleBits>
// void conv2d_bn_act(
// 	stream<ap_uint<Cin*Ibit> >& in, 
// 	const ap_uint<MVTU_InP*Wbit> weights[MVTU_OutP][((Cin*K*K)/MVTU_InP)*(Cout/MVTU_OutP)], 
// 	const ap_int<Mbit> factorA[MVTU_OutP][Cout/MVTU_OutP], 
// 	const ap_int<Mbit> factorB[MVTU_OutP][Cout/MVTU_OutP], 
// 	stream<ap_uint<Cout*Abit> >& out, 
// 	const unsigned reps = 1)
// {
// #pragma HLS DATAFLOW

// 	const unsigned Dout = Din/S + (Din%S > 0);
// 	const unsigned IntermediateDout = S*(Dout-1) + K;
// #ifdef CONV2_DEBUG
// 	cout << "Dout: " << Dout << endl;
// 	cout << "IntermediateDout: " << IntermediateDout << endl;
// #endif
// 	const unsigned TopLeftPad = (IntermediateDout - Din)/2;
// 	const unsigned BottomRightPad = (IntermediateDout - Din) - TopLeftPad;
// #ifdef CONV2_DEBUG
// 	cout << "TopLeftPad: " << TopLeftPad << endl;
// 	cout << "BottomRightPad: " << BottomRightPad << endl;
// #endif

// 	stream<ap_uint<Cin*Ibit> > samepad_out("samepad_out");
// 	SAMEPAD<TopLeftPad, BottomRightPad, Din, Cin, Ibit>(in, samepad_out, reps);
// #ifdef CONV2_DEBUG
// 	cout << "samepad_out.size(): " << samepad_out.size() << endl;
// #endif

// 	stream<ap_uint<Cin*Ibit> > swu_out("swu_out");
// 	SWU_NoP<K, S, IntermediateDout, Cin, Ibit> (samepad_out, swu_out, reps);

// 	stream<ap_uint<MVTU_InP*Ibit> > swu_out_reduced("swu_out_reduced");
// 	ReduceWidth<Cin*Ibit, MVTU_InP*Ibit, K*K*Dout*Dout> (swu_out, swu_out_reduced, reps);

// 	stream<ap_uint<MVTU_OutP*Abit> > out_raw("out_raw");
// 	MVAU_rowfirst<Dout*Dout, Ibit, Wbit, Mbit, Abit, Cin*K*K, Cout, MVTU_InP, MVTU_OutP, ScaleBits, FactorScaleBits>
// 	(swu_out_reduced, weights, factorA, factorB, out_raw, reps);
// #ifdef CONV2_DEBUG
// 	cout << "out_raw.size(): " << out_raw.size() << endl;
// #endif

// 	ExpandWidth<MVTU_OutP*Abit, Cout*Abit, Dout*Dout>
// 	(out_raw, out, reps);
// }

/**
 * 卷积计算单元
 * 
 */
template <	unsigned K,
			unsigned S,
			unsigned P,

			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,

			unsigned W_BIT,
			unsigned M_BIT,
			unsigned SIMD,
			unsigned PE>
void conv2d(
	stream<ap_uint<IN_CH*IN_BIT> >& in, 
	const ap_uint<SIMD*W_BIT> weights[PE][((IN_CH*K*K)/SIMD)*(OUT_CH/PE)],
	stream<ap_uint<OUT_CH*M_BIT> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	// const unsigned Dout = Din/S + (Din%S > 0);
	// const unsigned IntermediateDout = S*(Dout-1) + K;

	// 以下参数只适配 S = 1 时
	const unsigned INTER_ROW = IN_ROW + 2 * P;
	const unsigned INTER_COL = IN_COL + 2 * P;

	// 暂时认为输入 输出维度不变
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;


	stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
	
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, P>(in, padding_out, reps);


	stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
	SWU<K, S, INTER_ROW, INTER_COL, IN_CH, IN_BIT> (padding_out, swu_out, reps);
	// sliding_window_unit<K, S, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);

	//////////////test/////////////////////
	// stream<ap_uint<IN_BIT> > test_stream("test_stream");
	// adjust_width<IN_CH*IN_BIT, IN_BIT, K*K*OUT_ROW*OUT_COL>(swu_out, test_stream);

	// int test_res = 0;
	// for(int i=0; i < 9 * 2; i ++) {
		
	// 		for(int k=0; k < 16; k ++) {
	// 			ap_int<W_BIT> w = weights[0][i]((k+1)*2 -1, k*2);
	// 			ap_uint<IN_BIT> d = test_stream.read();
	// 			test_res += w * d;
	// 			// cout << d << "  ";
	// 		}
	// 		cout << test_res << "  ";
	// 		cout << "\n";
	// }

	// cout << "test_res = " << test_res;

	// 数据正确
	// 计算结果正确 应该 是矩阵向量乘单元存在问题
	//////////////test ////////////////////


	stream<ap_uint<SIMD*IN_BIT> > adj_out("adj_out");
	adjust_width<IN_CH*IN_BIT, SIMD*IN_BIT, K*K*OUT_ROW*OUT_COL>(swu_out, adj_out);

	stream<ap_uint<PE*M_BIT> > out_raw("out_raw");
	matrix_vector_unit<IN_CH*K*K, OUT_CH, IN_BIT, W_BIT, M_BIT, SIMD, PE, OUT_ROW*OUT_COL>(adj_out, weights, out_raw);

	adjust_width<PE*M_BIT, OUT_CH*M_BIT, OUT_ROW*OUT_COL*OUT_CH/PE>(out_raw, out);
}


/**
 * 卷积计算单元 同时计算bn_层与激活层
 * 使用MVTU在矩阵向量计算后立即计算得到激活输出值
 */
template <	unsigned K,
			unsigned S,
			unsigned P,

			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
			unsigned OUT_BIT,		// 量化激活后的位宽

			unsigned W_BIT,
			unsigned M_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,

			unsigned SIMD,
			unsigned PE>
void conv2d_bn_act(
	stream<ap_uint<IN_CH*IN_BIT> >& in, 
	const ap_uint<SIMD*W_BIT> weights[PE][((IN_CH*K*K)/SIMD)*(OUT_CH/PE)],
	const ap_uint<INC_BIT> inc[PE][OUT_CH/PE],
	const ap_int<BIAS_BIT> bias[PE][OUT_CH/PE],
	stream<ap_uint<OUT_CH*OUT_BIT> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	// const unsigned Dout = Din/S + (Din%S > 0);
	// const unsigned IntermediateDout = S*(Dout-1) + K;

	// 以下参数只适配 S = 1 时
	const unsigned INTER_ROW = IN_ROW + 2 * P;
	const unsigned INTER_COL = IN_COL + 2 * P;

	// 暂时认为输入 输出维度不变
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;


	// pading
	stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
	padding<IN_ROW, IN_COL, IN_CH, IN_BIT, P>(in, padding_out, reps);

	// 滑动窗口
	stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
	SWU<K, S, INTER_ROW, INTER_COL, IN_CH, IN_BIT> (padding_out, swu_out, reps);

	// 位宽调整
	stream<ap_uint<SIMD*IN_BIT> > adj_out("adj_out");
	adjust_width<IN_CH*IN_BIT, SIMD*IN_BIT, K*K*OUT_ROW*OUT_COL>(swu_out, adj_out);

	// 矩阵向量计算
	stream<ap_uint<PE*OUT_BIT> > out_raw("out_raw");
	matrix_vector_act_unit<IN_CH*K*K, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, OUT_ROW*OUT_COL>(adj_out, weights, inc, bias, out_raw);

    // // cout << "out_raw.size = " << out_raw.size() << "  ";
	adjust_width<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW*OUT_COL*OUT_CH/PE>(out_raw, out);
}
