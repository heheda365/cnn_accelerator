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

/**
 * 实现量化激活算法
 * 使用二分查找
 */
template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT>
ap_uint<OUT_BIT> bn_qurelu( ap_int<IN_BIT> in,
                ap_uint<INC_BIT> inc,
                ap_int<BIAS_BIT> bias ) {   
    ap_int<IN_BIT> target = in + bias;
    ap_uint<OUT_BIT> index = 1 << (OUT_BIT - 1);

    ap_int<IN_BIT> mid = inc << (OUT_BIT - 1);

    for(int i=OUT_BIT-2; i >= 0; i --) {
        // TODO
        // 因为不能特别确定 IN_BIT 和 inc_BIT 关系 所以这里可能有精度损失
        ap_int<IN_BIT> inc_shift = inc << i;
        ap_int<IN_BIT> one_shift = 1 << i;
        if(target < mid) {
            mid -= inc_shift;
            index -= one_shift; 
        } else if(mid < target){
            mid += inc_shift;
            index += one_shift;
        }
    }
    if(target < mid) {
        index --;
    }
    return index;
}

/**
 * 批正则化 和 量化激活函数
 */
// template <	unsigned IN_BIT,
// 			unsigned OUT_BIT,
// 			unsigned INC_BIT,
// 			unsigned BIAS_BIT,
//             unsigned SHIFT>
// void bn_qurelu( ap_int<IN_BIT> in,
//                 ap_uint<INC_BIT> inc,
//                 ap_int<BIAS_BIT> bias ) 
// {
//     target = target + bias;
//     int index = 1 << (BIT - 1);
//     int mid = inc << (BIT - 1);
//     for(int i=BIT-2; i >= 0; i --) {
//         int inc_shift = inc << i;
//         int one_shift = 1 << i;
//         if(target < mid) {
//             mid -= inc_shift;
//             index -= one_shift; 
//         } else if(mid < target){
//             mid += inc_shift;
//             index += one_shift;
//         }
//     }
//     if(target < mid) {
//         index --;
//     }
//     return index;
// }

// template<int IN_CH, int IN_ROW, int IN_COL, int IN_BIT, int OUT_BIT>
// void conv_bn_qrelu(int in[IN_CH][IN_ROW][IN_COL], int out[IN_CH][IN_ROW][IN_COL], int w[IN_CH], int b[IN_CH]) {
    
//     for(int ch=0; ch < IN_CH; ch ++) {
//         for(int row=0; row < IN_ROW; row ++) {
//             for(int col=0; col < IN_COL; col ++) {
//                 out[ch][row][col] = qrelu_search<OUT_BIT>(in[ch][row][col], w[ch], b[ch]);
//             }
//         }
//     }
// }

// template<int LEN, int IN_BIT, int OUT_BIT>
// void linear_bn_qrelu(int in[LEN], int out[LEN], int w[LEN], int b[LEN]) {
//     for(int i=0; i < LEN; i ++) {
       
//         out[i] = qrelu_search<OUT_BIT>(in[i], w[i], b[i]);
//     }
// }