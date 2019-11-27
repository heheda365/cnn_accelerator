#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#include <assert.h>
#include "function.h"

template <	unsigned Wbit,
			unsigned Mbit,
			unsigned M2bit,
			unsigned Abit,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
ap_uint<Abit> ACTIVATE(
	ap_int<M2bit> value, 
	ap_int<Mbit> factorA,
	ap_int<Mbit> factorB)
{
#pragma HLS PIPELINE II=1
	const ap_uint<Abit> limit = (1 << Abit)-1;

	ap_uint<Abit> result = 0;

	ap_int<Mbit+M2bit> temp_result = factorA*value;

	if (Wbit > 1)
		temp_result = temp_result >> ScaleBits;

	temp_result = temp_result + factorB;

	ap_uint<1> remains = temp_result(FactorScaleBits-1, FactorScaleBits-1);
	temp_result = temp_result >> FactorScaleBits;

	if (temp_result < 0)
		result = 0;
	else if (temp_result >= limit)
		result = limit;
	else
		result = temp_result(Abit-1, 0) + remains;

	return result;
}


template <	unsigned Wbit,
			unsigned Ibit,
			unsigned Mbit,
			unsigned P>
ap_int<Mbit> DOT(
	ap_uint<P*Wbit> weights, 
	ap_uint<P*Ibit> in) 
{	
	ap_int<Mbit> accumulation = 0;

	for (unsigned p = 0; p < P; p++) {
#pragma HLS UNROLL
		ap_int<Mbit> result;

		if (Wbit == 1) {
			ap_uint<Ibit> temp = in( (p+1)*Ibit-1, p*Ibit );
			if (weights(p,p) == 0)
				result = temp;
			else
				result = -temp;
		}
		else {
			ap_int<Wbit> temp_w = weights( (p+1)*Wbit-1, p*Wbit );
			ap_int<Ibit+1> temp_in = in( (p+1)*Ibit-1, p*Ibit );
			result = temp_w*temp_in;
		}

		accumulation += result;
	}

	return accumulation;
}



template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void MVTU(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*Wbit> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)],
	const ap_int<Mbit> factorA[OutP][MatrixW/OutP],
	const ap_int<Mbit> factorB[OutP][MatrixW/OutP],
	stream<ap_uint<OutP*Abit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_int<Mbit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Abit> outBuf;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
		}
		else {
			tempVec = rowstore[wVec];
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*Wbit> tempMat = weights[p][index];

			ap_int<Mbit> acc = DOT<Wbit, Ibit, Mbit, InP>( tempMat, tempVec );

			if (wVec == 0)
				resultVec[p] = acc;
			else
				resultVec[p] += acc;

			outBuf( (p+1)*Abit-1, p*Abit ) = ACTIVATE<Wbit, Mbit, Mbit, Abit, ScaleBits, FactorScaleBits>(resultVec[p], factorA[p][wMat], factorB[p][wMat]);
		}

		if (wVec == InputFold-1){
			out.write(outBuf);
		}

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}


template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InP,
			unsigned OutP>
void MVU(
	stream<ap_uint<InP*Ibit> >& vec, 
	const ap_uint<InP*Wbit> weights[OutP][(MatrixH/InP)*(MatrixW/OutP)], 
	stream<ap_uint<OutP*Mbit> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MatrixH%InP == 0, "MatrixH mod InP is not 0" );
	static_assert( MatrixW%OutP == 0, "MatrixW mod OutP is not 0");

	const unsigned InputFold = MatrixH/InP;
	const unsigned OutputFold = MatrixW/OutP;

#ifdef MVU_DEBUG
	cout << "InputFold: " << InputFold << endl;
	cout << "OutputFold: " << OutputFold << endl;
#endif

	const unsigned totalReps = reps*NumVecs*InputFold*OutputFold;

	ap_uint<InP*Ibit> rowstore[InputFold];
#pragma HLS RESOURCE variable=rowstore core=RAM_2P_BRAM

	ap_uint<Mbit> resultVec[OutP];
#pragma HLS ARRAY_PARTITION variable=resultVec complete dim=0
	unsigned wVec = 0;
	unsigned wMat = 0;
	ap_uint<InP*Ibit> tempVec;
	ap_uint<OutP*Mbit> outBuf;

	unsigned index = 0;
	for (unsigned rep = 0; rep < totalReps; rep++) {
#pragma HLS PIPELINE II=1
		
		if (wMat == 0) {
			tempVec = vec.read();
			rowstore[wVec] = tempVec;
		}
		else {
			tempVec = rowstore[wVec];
		}

		index = wVec*OutputFold+wMat;
		for (unsigned p = 0; p < OutP; p++) {
#pragma HLS UNROLL
			ap_uint<InP*Wbit> tempMat = weights[p][index];
			
			ap_int<Mbit> acc = DOT<Wbit, Ibit, Mbit, InP>( tempMat, tempVec );

			// 这个if也不应该放在这里面
			if (wVec == 0)
				resultVec[p] = acc;
			else
				resultVec[p] += acc;
			// 放在这里似乎不是很合理 因为 resultVec还在累加中此时写入buff没有意义
			// 如果这里需要做激活处理 则会浪费过多的逻辑单元
			outBuf((p+1)*Mbit-1, p*Mbit) = resultVec[p];
		}

		if (wVec == InputFold-1)
			out.write(outBuf);

		if (wVec == InputFold-1) {
			wVec = 0;
			if (wMat == OutputFold-1)
				wMat = 0;
			else
				wMat++;
		}
		else
			wVec++;
	}
}


/**
 * 矩阵向量计算单元
 * 
 */
template <	unsigned MAT_ROW,		// 展开后的k × k × in_ch
			unsigned MAT_COL,		// 展开后的out_ch
			unsigned IN_BIT,
			unsigned W_BIT,
			unsigned M_BIT,			// 乘累加后的计算结果的值
			unsigned SIMD,
			unsigned PE,
			unsigned VECT_NUMS>
void matrix_vector_unit(
	stream<ap_uint<SIMD*IN_BIT> >& vec, 
	const ap_uint<SIMD*W_BIT> weights[PE][(MAT_ROW/SIMD)*(MAT_COL/PE)], 
	stream<ap_uint<PE*M_BIT> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MAT_ROW%SIMD == 0, "MAT_ROW mod SIMD is not 0" );
	static_assert( MAT_COL%PE == 0, "MAT_COL mod PE is not 0");

	const unsigned INPUT_FOLD = MAT_ROW/SIMD;
	const unsigned OUTPUT_FOLD = MAT_COL/PE;

	const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;

	// 需要保存一行数据
	ap_uint<SIMD*IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

	// 用来保存累加结果
	ap_uint<M_BIT> result_vec[PE];
#pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
	unsigned in_fold_cnt = 0;			// 输入折叠计数
	unsigned out_fold_cnt = 0;			// 输出折叠计数
	unsigned tile = 0;

	// 一次 读入的数据 需要保存 in_ch * k * k长度的数据
	ap_uint<SIMD*IN_BIT> temp_vec;
	// 累加结果 这里需要初始化为0 
	ap_int<M_BIT> acc[PE] = {0};

	// total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1
		
		// 这里是在第一次输出之前 就度完了数据，之后一直用
		// 在输出折叠第一次计算时读
		if (out_fold_cnt == 0) {
			temp_vec = vec.read();
			row_store[in_fold_cnt] = temp_vec;
		}
		else {
			temp_vec = row_store[in_fold_cnt];
		}

		// index = wVec*OutputFold+wMat;

// 		// 初始化累加结果
// 		if(in_fold_cnt == 0) {
// 			for(int p=0; p < PE; p ++) {
// #pragma HLS UNROLL
// 				acc[p] = 0;
// 			}
// 		}

		// 主要计算单元 这里用UNROLL展开 期望用单周期实现计算
		// PE 并行计算
		for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
			// 读 W 子块
			ap_uint<SIMD*W_BIT> temp_mat = weights[p][tile];
			// SIMD 并行
			acc[p] = DOT<W_BIT, IN_BIT, M_BIT, SIMD>( temp_mat, temp_vec );
		}

		// 计数逻辑 和输出处理
		tile ++;
		if(++ in_fold_cnt == INPUT_FOLD) {
			in_fold_cnt = 0;
			ap_uint<PE*M_BIT> out_buf;
			// PE 列计算完成 可以输出
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				out_buf((p+1)*M_BIT-1, p*M_BIT) = acc[p];
				acc[p] = 0;
			}
			out.write(out_buf);
			// 完整的一次矩阵向量计算
			if(++ out_fold_cnt == OUTPUT_FOLD) {
				out_fold_cnt = 0;
				tile = 0;
			}

		}
	}  // end for

}

/**
 * 矩阵向量计算单元
 * 同时进行量化激活处理
 */
template <	unsigned MAT_ROW,		// 展开后的k × k × in_ch
			unsigned MAT_COL,		// 展开后的out_ch

			unsigned IN_BIT,
			unsigned OUT_BIT,		// 

			unsigned W_BIT,
			unsigned M_BIT,			// 乘累加后的计算结果的值

			unsigned INC_BIT,		// 激活等差数列 的步长
			unsigned BIAS_BIT,		// 

			unsigned SIMD,
			unsigned PE,
			unsigned VECT_NUMS>
void matrix_vector_act_unit(
	stream<ap_uint<SIMD*IN_BIT> >& vec, 
	const ap_uint<SIMD*W_BIT> weights[PE][(MAT_ROW/SIMD)*(MAT_COL/PE)], 
	const ap_uint<INC_BIT> inc[PE][MAT_COL/PE],
	const ap_int<BIAS_BIT> bias[PE][MAT_COL/PE],
	stream<ap_uint<PE*M_BIT> >& out, 
	const unsigned reps = 1) 
{
	static_assert( MAT_ROW%SIMD == 0, "MAT_ROW mod SIMD is not 0" );
	static_assert( MAT_COL%PE == 0, "MAT_COL mod PE is not 0");

	const unsigned INPUT_FOLD = MAT_ROW/SIMD;
	const unsigned OUTPUT_FOLD = MAT_COL/PE;

	const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;

	// 需要保存一行数据
	ap_uint<SIMD*IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

	// 用来保存累加结果
	ap_uint<M_BIT> result_vec[PE];
#pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
	unsigned in_fold_cnt = 0;			// 输入折叠计数
	unsigned out_fold_cnt = 0;			// 输出折叠计数
	unsigned tile = 0;

	// 一次 读入的数据 需要保存 in_ch * k * k长度的数据
	ap_uint<SIMD*IN_BIT> temp_vec;
	// 累加结果 这里需要初始化为0 
	ap_int<M_BIT> acc[PE] = {0};

	// total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
	for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1
		
		// 这里是在第一次输出之前 就度完了数据，之后一直用
		// 在输出折叠第一次计算时读
		if (out_fold_cnt == 0) {
			temp_vec = vec.read();
			row_store[in_fold_cnt] = temp_vec;
		}
		else {
			temp_vec = row_store[in_fold_cnt];
		}

		// index = wVec*OutputFold+wMat;

// 		// 初始化累加结果
// 		if(in_fold_cnt == 0) {
// 			for(int p=0; p < PE; p ++) {
// #pragma HLS UNROLL
// 				acc[p] = 0;
// 			}
// 		}

		// 主要计算单元 这里用UNROLL展开 期望用单周期实现计算
		// PE 并行计算
		for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
			// 读 W 子块
			ap_uint<SIMD*W_BIT> temp_mat = weights[p][tile];
			// SIMD 并行
			acc[p] = DOT<W_BIT, IN_BIT, M_BIT, SIMD>( temp_mat, temp_vec );
		}

		// 计数逻辑 和输出处理
		tile ++;
		if(++ in_fold_cnt == INPUT_FOLD) {
			in_fold_cnt = 0;
			ap_uint<PE*M_BIT> out_buf;
			// PE 列计算完成 可以输出
			for(unsigned p=0; p < PE; p ++) {
#pragma HLS UNROLL
				out_buf((p+1)*M_BIT-1, p*M_BIT) = bn_qurelu<M_BIT, OUT_BIT, IN_BIT, BIAS_BIT>(acc[p], inc[p][out_fold_cnt], bias[p][out_fold_cnt]);
				acc[p] = 0;
			}
			out.write(out_buf);
			// 完整的一次矩阵向量计算
			if(++ out_fold_cnt == OUTPUT_FOLD) {
				out_fold_cnt = 0;
				tile = 0;
			}

		}
	}  // end for

}